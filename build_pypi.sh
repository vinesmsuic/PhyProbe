#!/usr/bin/env bash
# Build and optionally publish the phyprobe PyPI package.
#
# Usage:
#   ./build_pypi.sh              # build only (wheel + sdist)
#   ./build_pypi.sh --test       # upload to TestPyPI
#   ./build_pypi.sh --publish    # upload to PyPI

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="build"
for arg in "$@"; do
  case "$arg" in
    --test)    MODE="test" ;;
    --publish) MODE="publish" ;;
    *)         echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── 1. Clean previous artifacts ───────────────────────────────────────────────
echo "==> Cleaning dist/ and build/"
rm -rf dist/ build/ phyprobe.egg-info phyprobe/__pycache__

# ── 2. Ensure build tools are available ───────────────────────────────────────
echo "==> Checking build tools"
python -m pip install --quiet --upgrade pip build twine

# ── 3. Build wheel + sdist ────────────────────────────────────────────────────
echo "==> Building wheel and sdist"
python -m build

echo ""
echo "Built artifacts:"
ls -lh dist/

# ── 4. Check distributions ────────────────────────────────────────────────────
echo ""
echo "==> Checking distributions with twine"
python -m twine check dist/*

# ── 5. Upload (optional) ──────────────────────────────────────────────────────
if [[ "$MODE" == "test" ]]; then
  echo ""
  echo "==> Uploading to TestPyPI"
  python -m twine upload --repository testpypi dist/*
  echo ""
  echo "Install from TestPyPI with:"
  echo "  pip install --index-url https://test.pypi.org/simple/ phyprobe"

elif [[ "$MODE" == "publish" ]]; then
  echo ""
  read -r -p "Publish to PyPI? This cannot be undone. [y/N] " confirm
  if [[ "$confirm" =~ ^[Yy]$ ]]; then
    python -m twine upload dist/*
    echo ""
    echo "Published. Install with: pip install phyprobe"
  else
    echo "Aborted."
  fi

else
  echo ""
  echo "Build complete. To upload:"
  echo "  ./build_pypi.sh --test      # TestPyPI"
  echo "  ./build_pypi.sh --publish   # PyPI"
fi
