"""PostgreSQL connection helpers for the international AlphaEarth study.

Credentials come from environment variables only — never hardcode passwords.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Prefer project micromamba psycopg when available.
try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "psycopg is required. Use .tools/postgis/bin/python or pip install 'psycopg[binary]'."
    ) from exc


ENV_KEYS = (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
)


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no dependency). Does not override existing env vars."""
    if path is None:
        here = Path(__file__).resolve()
        candidates = [
            here.parents[1] / "db" / ".env",
            here.parents[3] / ".env",
        ]
    else:
        candidates = [path]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)
        break


def db_config() -> dict:
    load_dotenv()
    missing = [k for k in ENV_KEYS if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing database env vars: {missing}. "
            "Copy db/.env.example → db/.env and export them for jobs."
        )
    return {
        "host": os.environ["POSTGRES_HOST"],
        "port": int(os.environ["POSTGRES_PORT"]),
        "dbname": os.environ["POSTGRES_DB"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
    }


def dsn() -> str:
    c = db_config()
    # Password is only in the DSN string in-process; do not print this.
    return (
        f"host={c['host']} port={c['port']} dbname={c['dbname']} "
        f"user={c['user']} password={c['password']}"
    )


@contextmanager
def connect(*, autocommit: bool = False) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(dsn(), row_factory=dict_row, autocommit=autocommit)
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def wait_until_ready(timeout_s: float = 60.0, interval_s: float = 1.0) -> None:
    import time

    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with connect(autocommit=True) as conn:
                conn.execute("SELECT 1")
            return
        except Exception as err:  # noqa: BLE001 — retry until timeout
            last_err = err
            time.sleep(interval_s)
    raise TimeoutError(f"Database not reachable within {timeout_s}s: {last_err}")
