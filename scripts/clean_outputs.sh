#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Cleaning generated outputs and caches..."

mkdir -p data/raw data/processed reports

touch data/raw/.gitkeep data/processed/.gitkeep reports/.gitkeep

# Remove local DB and SQLite artifacts
find data -type f \( -name "*.db" -o -name "*.sqlite" -o -name "*.sqlite3" -o -name "*.sqlite*" \) -delete 2>/dev/null || true

# Remove generated raw and processed outputs but keep folder placeholders
find data/raw -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} + 2>/dev/null || true
find data/processed -type f ! -name ".gitkeep" -delete 2>/dev/null || true

# Remove generated reports but keep placeholder
find reports -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} + 2>/dev/null || true

# Remove Python and tooling caches
find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
rm -rf .pytest_cache .mypy_cache .ruff_cache

# Remove macOS metadata files
find . -type f -name ".DS_Store" -delete 2>/dev/null || true

echo "Cleanup complete."
