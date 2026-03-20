#!/bin/bash

REPO_PATH="/Users/janikwahrheit/Library/CloudStorage/OneDrive-Persönlich/01_Studium/01_Bachelor/Bachelorarbeit/Code"

LFS_THRESHOLD=$((95 * 1024 * 1024))        # 95 MB
GITHUB_LIMIT=$((100 * 1024 * 1024))       # 100 MB

SCRIPT_NAME="git_lfs_auto.sh"

cd "$REPO_PATH" || { echo "❌ Repo nicht gefunden"; exit 1; }

echo
echo "🔍 Scanne Dateien in $REPO_PATH"
echo "--------------------------------------------------"


touch .gitignore

LFS_FILES=()
GITHUB_BLOCKERS=()


while IFS= read -r file; do
    [ "$file" = "./$SCRIPT_NAME" ] && continue


    filesize=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")

    [ -z "$filesize" ] && continue

   
    clean_file="${file#./}"
    
    size_mb=$(bc <<< "scale=2; $filesize/1024/1024")

    printf "%8.2f MB  %s\n" "$size_mb" "$clean_file"


    if [ "$filesize" -ge "$LFS_THRESHOLD" ]; then
        LFS_FILES+=("$clean_file ($size_mb MB)")
        

        if ! grep -qxF "$clean_file" .gitignore; then
            echo "$clean_file" >> .gitignore
      
            git rm --quiet --cached "$clean_file" 2>/dev/null
        fi
    fi

    if [ "$filesize" -ge "$GITHUB_LIMIT" ]; then
        GITHUB_BLOCKERS+=("$clean_file ($size_mb MB)")
    fi
done < <(find . -type f -not -path "*/\.git/*")

echo
echo "=================================================="
echo "📦 Dateien über LFS-Threshold (>95 MB) -> In .gitignore"
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
echo "🚨 Dateien über GitHub-Limit (>100 MB) -> In .gitignore"
echo "=================================================="

if [ ${#GITHUB_BLOCKERS[@]} -eq 0 ]; then
    echo "✅ Keine"
else
    for f in "${GITHUB_BLOCKERS[@]}"; do
        echo "❌  $f"
    done
fi

echo
echo "✅ Alle großen Dateien sind jetzt sicher in der .gitignore!"