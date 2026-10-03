#!/usr/bin/env bash
# The Warp backend on a Jetson: Warp, MuJoCo and MuJoCo Warp, no JAX.
# Run from the repository root.
set -eu
uv sync --locked --no-dev
./scripts/fetch_menagerie.sh
uv run --locked --no-dev python deploy/orin/check.py
