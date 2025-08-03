#!/usr/bin/env bash

# Check for T O D O ! patterns that shouldn't be committed
#
# This hook fails if any file contains "T O D O !" (without spaces) which is used to mark
# temporary changes that should not be committed to the repository.
set -e

# If no files are passed as arguments, check all staged files
if [ $# -eq 0 ]; then
    # Get list of staged files
    files=$(git diff --cached --name-only --diff-filter=ACM | grep -v -E '\.(md|yaml|yml)$' || true)
else
    # Use the files passed as arguments (for --all-files mode)
    files="$@"
fi

# If no files to check, exit successfully
if [ -z "$files" ]; then
    exit 0
fi

# Search for T O D O ! (without spaces) in the files
matches=""
for file in $files; do
    if [ -f "$file" ]; then
        # Look for T O D O ! pattern (concatenated to avoid self-match)
        pattern="TO""DO!"
        file_matches=$(grep -n "$pattern" "$file" 2>/dev/null || true)
        if [ -n "$file_matches" ]; then
            matches="${matches}${file}:${file_matches}\n"
        fi
    fi
done

if [[ -n "$matches" ]]; then
  # Count the number of matches to use proper grammar
  count=$(echo "$matches" | wc -l)
  if [[ $count -eq 1 ]]; then
    echo "T O D O ! marker detected (should not be committed):"
    echo "$matches"
    echo ""
    echo "Please resolve this T O D O ! marker before committing."
  else
    echo "T O D O ! markers detected (should not be committed):"
    echo "$matches"
    echo ""
    echo "Please resolve these T O D O ! markers before committing."
  fi
  exit 1
fi
