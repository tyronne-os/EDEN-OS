#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# EDEN OS — Versioned Save Protocol
# Saves to: GitHub, HuggingFace, Seagate 5TB (with version tags)
#
# Usage: bash scripts/save_versioned.sh [version_tag]
# Example: bash scripts/save_versioned.sh v1.0.1
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/.."

VERSION="${1:-$(date +v%Y%m%d_%H%M%S)}"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

echo "═══════════════════════════════════════════════════"
echo " EDEN OS — VERSIONED SAVE: $VERSION"
echo " $TIMESTAMP"
echo "═══════════════════════════════════════════════════"

# ── 1. Git Commit + Tag ──────────────────────────────────────
echo ""
echo "[1/4] Git commit + tag..."
git add -A
git commit -m "EDEN OS $VERSION — $TIMESTAMP" --allow-empty 2>/dev/null || echo "Nothing to commit"
git tag -a "$VERSION" -m "EDEN OS $VERSION — $TIMESTAMP" 2>/dev/null || echo "Tag $VERSION already exists"

# ── 2. Push to GitHub ────────────────────────────────────────
echo ""
echo "[2/4] Pushing to GitHub (tyronne-os/EDEN-OS)..."
git push origin main --tags 2>&1 || echo "GitHub push failed (check connectivity)"

# ── 3. Push to HuggingFace ───────────────────────────────────
echo ""
echo "[3/4] Uploading to HuggingFace (AIBRUH/eden-os)..."
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='.',
    repo_id='AIBRUH/eden-os',
    repo_type='model',
    ignore_patterns=['.venv/*', '__pycache__/*', '*.pyc', 'models_cache/*', 'data/*', '.git/*'],
    commit_message='EDEN OS $VERSION',
)
print('HuggingFace upload complete')
" 2>&1 || echo "HuggingFace upload failed"

# ── 4. Save to Seagate 5TB (versioned) ───────────────────────
echo ""
echo "[4/4] Saving to Seagate 5TB..."

SEAGATE_PATH=""
# Detect Seagate
for path in /mnt/s /mnt/S /mnt/seagate5tb; do
  if [ -d "$path" ] && touch "$path/.eden_test" 2>/dev/null; then
    rm "$path/.eden_test"
    SEAGATE_PATH="$path"
    break
  fi
done

# Try PowerShell detection
if [ -z "$SEAGATE_PATH" ]; then
  LETTER=$(powershell.exe -Command "(Get-Volume -FileSystemLabel 'SEAGATE5TB').DriveLetter" 2>/dev/null | tr -d '\r\n')
  if [ -n "$LETTER" ]; then
    WSL_PATH="/mnt/$(echo $LETTER | tr 'A-Z' 'a-z')"
    if [ -d "$WSL_PATH" ]; then
      SEAGATE_PATH="$WSL_PATH"
    else
      sudo mkdir -p /mnt/seagate5tb 2>/dev/null
      sudo mount -t drvfs "${LETTER}:" /mnt/seagate5tb 2>/dev/null && SEAGATE_PATH="/mnt/seagate5tb"
    fi
  fi
fi

if [ -n "$SEAGATE_PATH" ]; then
  BACKUP_DIR="$SEAGATE_PATH/eden-os/versions/$VERSION"
  LATEST_DIR="$SEAGATE_PATH/eden-os/latest"

  echo "Seagate detected at: $SEAGATE_PATH"

  # Create versioned backup
  mkdir -p "$BACKUP_DIR"
  rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='models_cache' --exclude='data' --exclude='.git' \
    . "$BACKUP_DIR/"

  # Update latest symlink
  rm -f "$LATEST_DIR" 2>/dev/null
  ln -sf "$BACKUP_DIR" "$LATEST_DIR"

  # Write version manifest
  cat > "$SEAGATE_PATH/eden-os/VERSION_LOG.md" << MANIFEST
# EDEN OS — Version Log (Seagate 5TB)
Last updated: $TIMESTAMP
Latest version: $VERSION

## Versions on disk:
$(ls -1d "$SEAGATE_PATH/eden-os/versions/"* 2>/dev/null | while read d; do
  V=$(basename "$d")
  SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
  echo "- $V ($SIZE)"
done)
MANIFEST

  echo "Saved to Seagate: $BACKUP_DIR"
  echo "Latest symlink: $LATEST_DIR"

  # Cleanup: keep only last 10 versions
  VERSIONS_DIR="$SEAGATE_PATH/eden-os/versions"
  VERSION_COUNT=$(ls -1d "$VERSIONS_DIR/"* 2>/dev/null | wc -l)
  if [ "$VERSION_COUNT" -gt 10 ]; then
    echo "Cleaning old versions (keeping last 10)..."
    ls -1td "$VERSIONS_DIR/"* | tail -n +11 | while read old; do
      echo "  Removing: $(basename $old)"
      rm -rf "$old"
    done
  fi
else
  echo "Seagate 5TB not connected — skipping local backup"
  echo "Plug in Seagate and re-run: bash scripts/save_versioned.sh $VERSION"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo " EDEN OS $VERSION — SAVED"
echo " GitHub:      tyronne-os/EDEN-OS"
echo " HuggingFace: AIBRUH/eden-os"
echo " Seagate:     $([ -n "$SEAGATE_PATH" ] && echo "$SEAGATE_PATH/eden-os/versions/$VERSION" || echo "NOT CONNECTED")"
echo " OWN THE SCIENCE."
echo "═══════════════════════════════════════════════════"
