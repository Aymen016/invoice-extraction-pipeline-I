#!/usr/bin/env bash
# Starts the API and the review UI together, and shuts both down on Ctrl-C.
set -euo pipefail

cleanup() { kill 0; }
trap cleanup EXIT INT TERM

echo "API  → http://localhost:8000/docs"
echo "UI   → http://localhost:5173"
echo

uvicorn src.api:app --reload --port 8000 &
(cd frontend && npm run dev) &
wait
