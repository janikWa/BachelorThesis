#!/bin/bash

REPO_PATH="/Users/janikwahrheit/Library/CloudStorage/OneDrive-Persönlich/01_Studium/01_Bachelor/Bachelorarbeit/Code"

LFS_THRESHOLD=$((95 * 1024 * 1024))        # 95 MB
GITHUB_LIMIT=$((100 * 1024 * 1024))       # 100 MB

SCRIPT_NAME="git_lfs_auto.sh"

cd "$REPO_PATH" || { echo "❌ Repo nicht gefunden"; exit 1; }

git lfs install --quiet

echo
echo "🔍 Scanne Dateien in $REPO_PATH"
echo "--------------------------------------------------"

LFS_FILES=()
GITHUB_BLOCKERS=()

while IFS= read -r file; do
    [ "$file" = "./$SCRIPT_NAME" ] && continue

    # macOS / Linux kompatibel
    filesize=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")

    printf "%8.2f MB  %s\n" "$(bc <<< "scale=2; $filesize/1024/1024")" "$file"

    if [ "$filesize" -ge "$LFS_THRESHOLD" ]; then
        LFS_FILES+=("$file")
    fi

    if [ "$filesize" -ge "$GITHUB_LIMIT" ]; then
        GITHUB_BLOCKERS+=("$file")
    fi
done < <(find . -type f)

echo
echo "=================================================="
echo "📦 Dateien über LFS-Threshold (>95 MB)"
echo "=================================================="

if [ ${#LFS_FILES[@]} -eq 0 ]; then
    echo "✅ Keine"
else
    for f in "${LFS_FILES[@]}"; do
        echo "➡️  $f"
    done
fi

echo
echo "=================================================="
echo "🚨 Dateien über GitHub-Limit (>100 MB)"
echo "=================================================="

if [ ${#GITHUB_BLOCKERS[@]} -eq 0 ]; then
    echo "✅ Keine"
else
    for f in "${GITHUB_BLOCKERS[@]}"; do
        echo "❌  $f"
    done

