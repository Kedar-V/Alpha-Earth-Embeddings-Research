#!/usr/bin/env python3
"""Migrate this workspace to kv133@pizer.egr.duke.edu via rsync + tqdm.

Designed to be driven from notebooks/migrate_workspace_to_pizer.ipynb
(password via getpass). Also usable as a CLI:

  python scripts/migrate_to_pizer.py --prompt-password
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import shlex
import stat as stat_mod
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pexpect
from tqdm.auto import tqdm

DEFAULT_USER = "kv133"
DEFAULT_HOST = "pizer.egr.duke.edu"
DEFAULT_PORT = 22
DEFAULT_SRC = Path("/data/users/kv133/Alpha-Earth-Embeddings-Research")
REMOTE_CANDIDATES = [
    "/data/users/kv133/Alpha-Earth-Embeddings-Research",
    "/home/kv133/Alpha-Earth-Embeddings-Research",
]

# rsync --info=progress2 line, e.g.
#   1,072,881,664  35%  119.75MB/s    0:00:47 (xfr#1238, to-chk=4521/5760)
_PROGRESS_RE = re.compile(
    r"^\s*([\d,]+)\s+(\d+)%\s+([\d.]+[KkMmGgTt]?B/s)\s+"
    r"([\d:]+)\s*(?:\(xfr#(\d+),\s*to-chk=(\d+)/(\d+)\))?"
)
_PASSWORD_RE = re.compile(r"[Pp]assword:")


@dataclass
class MigrateConfig:
    src: Path = DEFAULT_SRC
    remote_user: str = DEFAULT_USER
    remote_host: str = DEFAULT_HOST
    remote_port: int = DEFAULT_PORT
    remote_dest: str | None = None  # auto-pick if None
    excludes: tuple[str, ...] = ()
    dry_run: bool = False
    password: str | None = None


def _ssh_opts(
    control_path: str | None = None,
    *,
    master: bool = False,
    port: int = DEFAULT_PORT,
) -> list[str]:
    opts = [
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "NumberOfPasswordPrompts=1",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=6",
        "-p",
        str(port),
    ]
    if control_path:
        opts += [
            "-o",
            f"ControlPath={control_path}",
            "-o",
            "ControlMaster=yes" if master else "ControlMaster=auto",
            "-o",
            "ControlPersist=900",
        ]
    return opts


def _feed_password(child: pexpect.spawn, password: str, timeout: float = 90) -> None:
    idx = child.expect([_PASSWORD_RE, pexpect.EOF, pexpect.TIMEOUT], timeout=timeout)
    if idx == 0:
        child.sendline(password)
        return
    if idx == 1:
        # Already authenticated / command finished without prompt
        return
    raise TimeoutError("timed out waiting for SSH password prompt")


def ssh_run(
    remote_cmd: str,
    *,
    password: str,
    user: str = DEFAULT_USER,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    control_path: str | None = None,
    timeout: float = 180,
    check: bool = True,
) -> str:
    cmd = ["ssh", *_ssh_opts(control_path, port=port), f"{user}@{host}", remote_cmd]
    child = pexpect.spawn(
        cmd[0], cmd[1:], encoding="utf-8", timeout=timeout, maxread=1 << 20
    )
    _feed_password(child, password, timeout=min(90, timeout))
    child.expect(pexpect.EOF, timeout=timeout)
    out = child.before or ""
    child.close()
    status = child.exitstatus
    # pexpect sometimes leaves exitstatus None; treat that as unknown/ok
    if check and status not in (0, None):
        raise RuntimeError(f"ssh exit {status}\ncmd={cmd!r}\nout:\n{out}")
    return out


def open_control_master(
    password: str,
    *,
    user: str = DEFAULT_USER,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    control_path: str,
) -> None:
    cmd = [
        "ssh",
        "-MNf",
        *_ssh_opts(control_path, master=True, port=port),
        f"{user}@{host}",
    ]
    child = pexpect.spawn(cmd[0], cmd[1:], encoding="utf-8", timeout=90)
    _feed_password(child, password)
    child.expect(pexpect.EOF, timeout=90)
    child.close()
    check = subprocess.run(
        ["ssh", *_ssh_opts(control_path, port=port), "-O", "check", f"{user}@{host}"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            f"SSH control master failed:\n{check.stderr or check.stdout}\n"
            f"(exit {check.returncode})"
        )


def close_control_master(
    *,
    user: str = DEFAULT_USER,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    control_path: str,
) -> None:
    subprocess.run(
        ["ssh", *_ssh_opts(control_path, port=port), "-O", "exit", f"{user}@{host}"],
        capture_output=True,
        text=True,
    )


def _first_abs_path_line(text: str) -> str | None:
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("/") and " " not in s and len(s) < 512:
            return s
    return None


def pick_remote_dest(password: str, cfg: MigrateConfig) -> str:
    if cfg.remote_dest:
        return cfg.remote_dest
    # Prefer /data/users/... when parent is writable; else ~/.
    # Avoid `bash -lc` + `set -e` (login profile can force exit 1 even after a
    # successful echo). Use a clean non-login bash and accept stdout path even
    # if ssh reports a non-zero status.
    body_parts: list[str] = []
    for p in REMOTE_CANDIDATES:
        parent = shlex.quote(str(Path(p).parent))
        dest = shlex.quote(p)
        body_parts.append(
            f"if [ -d {parent} ] && [ -w {parent} ]; then echo {dest}; exit 0; fi"
        )
    body_parts.append('echo "$HOME/Alpha-Earth-Embeddings-Research"; exit 0')
    remote = "bash --noprofile --norc -c " + shlex.quote("; ".join(body_parts))
    try:
        out = ssh_run(
            remote,
            password=password,
            user=cfg.remote_user,
            host=cfg.remote_host,
            port=cfg.remote_port,
            check=True,
        )
    except RuntimeError as exc:
        # Command often still printed the path; recover it from the error text.
        out = str(exc)
    dest = _first_abs_path_line(out)
    if not dest:
        raise RuntimeError(f"could not pick remote dest; ssh out:\n{out}")
    return dest


def remote_disk_report(password: str, cfg: MigrateConfig, dest: str) -> str:
    parent = str(Path(dest).parent)
    return ssh_run(
        f"hostname; df -h {shlex.quote(parent)} ~ 2>/dev/null; "
        f"mkdir -p {shlex.quote(dest)}; du -sh {shlex.quote(dest)} 2>/dev/null || true",
        password=password,
        user=cfg.remote_user,
        host=cfg.remote_host,
        port=cfg.remote_port,
        timeout=120,
    )


def local_size_bytes(src: Path, excludes: Sequence[str] = ()) -> int:
    """Approx on-disk bytes (like `du`), not sparse logical size.

    Uses ``st_blocks * 512`` so Postgres/numpy sparse holes are not counted
    as full ``st_size``. Dedupes hardlinks by inode.
    """
    excl = set(excludes)
    excl_names = {e.rstrip("/") for e in excl}
    total = 0
    seen: set[tuple[int, int]] = set()
    for root, dirs, files in os.walk(src, followlinks=False):
        pruned = []
        for d in list(dirs):
            rel = str((Path(root) / d).relative_to(src)).replace("\\", "/") + "/"
            if d + "/" in excl or rel in excl or d in excl_names:
                pruned.append(d)
        for d in pruned:
            dirs.remove(d)
        for f in files:
            fp = Path(root) / f
            try:
                st = fp.lstat()
            except OSError:
                continue
            if stat_mod.S_ISLNK(st.st_mode):
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_blocks * 512
    return total


def _parse_progress(line: str) -> dict | None:
    m = _PROGRESS_RE.search(line.replace("\r", "").strip())
    if not m:
        return None
    transferred = int(m.group(1).replace(",", ""))
    return {
        "bytes": transferred,
        "pct": int(m.group(2)),
        "rate": m.group(3),
        "eta": m.group(4),
        "xfr": int(m.group(5)) if m.group(5) else None,
        "to_chk": int(m.group(6)) if m.group(6) else None,
        "to_chk_total": int(m.group(7)) if m.group(7) else None,
    }


def rsync_with_tqdm(
    cfg: MigrateConfig,
    *,
    password: str,
    dest: str,
    control_path: str,
    total_bytes: int | None = None,
) -> int:
    """Run rsync over an existing SSH control master. Returns exit code."""
    # Prefer RSYNC_RSH over -e: long -e strings get awkward in ps/argv.
    ssh_e = "ssh " + " ".join(
        shlex.quote(x) for x in _ssh_opts(control_path, port=cfg.remote_port)
    )
    # Incremental recursion (default): start sending files immediately.
    # --no-inc-recursive looked "stuck" for minutes while scanning ~100k+ files
    # with the tqdm bar frozen near 0.
    cmd: list[str] = [
        "rsync",
        "-aH",
        "--numeric-ids",
        "--info=progress2",
        "--partial",
        "--partial-dir=.rsync-partial",
        "--human-readable",
        "--timeout=120",
    ]
    if cfg.dry_run:
        cmd.append("--dry-run")
    for ex in cfg.excludes:
        cmd += ["--exclude", ex]
    # trailing slash: copy contents into dest
    src = str(cfg.src).rstrip("/") + "/"
    remote = f"{cfg.remote_user}@{cfg.remote_host}:{dest.rstrip('/')}/"
    cmd += [src, remote]

    env = os.environ.copy()
    env["RSYNC_RSH"] = ssh_e

    print(
        "starting rsync (incremental — files should move within seconds; "
        "bar may jump as totals refine) …",
        flush=True,
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None

    bar = tqdm(
        total=total_bytes if total_bytes and total_bytes > 0 else None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="rsync → pizer",
        dynamic_ncols=True,
        mininterval=0.3,
    )
    file_bar = tqdm(
        total=None,
        unit="file",
        desc="files xfr'd",
        dynamic_ncols=True,
        position=1,
        leave=True,
        mininterval=0.5,
    )

    last_bytes = 0
    last_xfr = 0
    buf = ""
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            # progress2 uses \r updates
            buf += chunk
            parts = re.split(r"[\r\n]", buf)
            buf = parts[-1]
            for line in parts[:-1]:
                line = line.strip()
                if not line:
                    continue
                prog = _parse_progress(line)
                if not prog:
                    # occasional non-progress lines
                    if "rsync error" in line.lower() or "permission denied" in line.lower():
                        tqdm.write(line)
                    continue
                b = prog["bytes"]
                delta = max(0, b - last_bytes)
                if bar.total:
                    bar.update(delta)
                else:
                    bar.n = b
                    bar.refresh()
                last_bytes = b
                bar.set_postfix_str(
                    f"{prog['pct']}% {prog['rate']} eta {prog['eta']}", refresh=False
                )
                if prog["xfr"] is not None:
                    dx = max(0, prog["xfr"] - last_xfr)
                    file_bar.update(dx)
                    last_xfr = prog["xfr"]
                    if prog["to_chk_total"]:
                        file_bar.set_postfix_str(
                            f"to-chk {prog['to_chk']}/{prog['to_chk_total']}",
                            refresh=False,
                        )
        rc = proc.wait()
    finally:
        bar.close()
        file_bar.close()

    if rc != 0:
        raise RuntimeError(f"rsync failed with exit code {rc}")
    return rc


def migrate(cfg: MigrateConfig) -> dict:
    if not cfg.password:
        raise ValueError("password required")
    src = cfg.src.resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)

    print(f"local src: {src}")
    print("probing remote …")
    dest = pick_remote_dest(cfg.password, cfg)
    print(f"remote dest: {cfg.remote_user}@{cfg.remote_host}:{dest}")
    print(remote_disk_report(cfg.password, cfg, dest))

    print("estimating local size (may take a minute) …")
    total = local_size_bytes(src, cfg.excludes)
    print(f"approx local bytes: {total:,} ({total / (1024**3):.2f} GiB)")

    with tempfile.TemporaryDirectory(prefix="pizer_ssh_") as td:
        control_path = str(Path(td) / "cm-%r@%h:%p")
        print("opening SSH control master (one password prompt) …")
        open_control_master(
            cfg.password,
            user=cfg.remote_user,
            host=cfg.remote_host,
            port=cfg.remote_port,
            control_path=control_path,
        )
        try:
            # ensure dest exists via multiplexed socket (no password)
            subprocess.run(
                [
                    "ssh",
                    *_ssh_opts(control_path, port=cfg.remote_port),
                    f"{cfg.remote_user}@{cfg.remote_host}",
                    f"mkdir -p {shlex.quote(dest)}",
                ],
                check=True,
            )
            t0 = time.time()
            rsync_with_tqdm(
                cfg,
                password=cfg.password,
                dest=dest,
                control_path=control_path,
                total_bytes=total,
            )
            dt = time.time() - t0
            print(f"rsync finished in {dt / 60:.1f} min")
        finally:
            close_control_master(
                user=cfg.remote_user,
                host=cfg.remote_host,
                port=cfg.remote_port,
                control_path=control_path,
            )

    # verify
    verify = ssh_run(
        f"du -sh {shlex.quote(dest)}; "
        f"test -d {shlex.quote(dest + '/.git')} && echo HAS_GIT=yes; "
        f"ls {shlex.quote(dest)} | head",
        password=cfg.password,
        user=cfg.remote_user,
        host=cfg.remote_host,
        port=cfg.remote_port,
    )
    print("remote verify:\n", verify)
    return {"dest": dest, "approx_bytes": total, "verify": verify}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dest", default=None, help="remote absolute path")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--prompt-password", action="store_true")
    args = p.parse_args(argv)

    password = None
    if args.prompt_password or sys.stdin.isatty():
        password = getpass.getpass(f"Password for {args.user}@{args.host}: ")
    else:
        password = os.environ.get("PIZER_PASSWORD")
    if not password:
        print("Set PIZER_PASSWORD or pass --prompt-password", file=sys.stderr)
        return 2

    cfg = MigrateConfig(
        src=args.src,
        remote_user=args.user,
        remote_host=args.host,
        remote_dest=args.dest,
        excludes=tuple(args.exclude),
        dry_run=args.dry_run,
        password=password,
    )
    migrate(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
