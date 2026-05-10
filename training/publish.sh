#!/usr/bin/env bash
# Copy the latest trained weights into public/weights/expert.json so the web
# app picks them up. Safe to run while training is in progress.
set -euo pipefail
cd "$(dirname "$0")/.."
src="training/runs/expert/weights-latest.json"
dst="public/weights/expert.json"
if [ ! -f "$src" ]; then
  echo "no weights yet at $src" >&2
  exit 1
fi
mkdir -p "$(dirname "$dst")"
cp "$src" "$dst"
sz=$(wc -c < "$dst")
echo "published $dst (${sz} bytes)"
