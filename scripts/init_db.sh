#!/usr/bin/env bash
# Repo-root convenience wrapper.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/week4/international_geo_generalization/scripts/init_db.sh" "$@"
