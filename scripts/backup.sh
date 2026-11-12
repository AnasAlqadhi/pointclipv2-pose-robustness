#!/bin/bash
# Back up everything that is expensive to recreate, to OneDrive.
# Not backed up: data/ (2.2 GB, re-downloadable), third_party/ (a pinned clone),
# .venv/ (rebuildable from SETUP.md). Those are recorded, not copied.
set -e
SRC="/d/Projects/school/courses/2026-fall-YZM5110-multimodal-learning"
DEST="/c/Users/anasm/OneDrive/Backups/YZM5110-pointclipv2"
mkdir -p "$DEST"
for d in project/results project/docs project/src project/scripts project/configs project/tests; do
  mkdir -p "$DEST/$d"
  cp -r "$SRC/$d/." "$DEST/$d/" 2>/dev/null || true
done
cp "$SRC"/*.md "$DEST/" 2>/dev/null || true
cp "$SRC/project"/*.md "$SRC/project/requirements.txt" "$DEST/project/" 2>/dev/null || true
cp "$SRC/files/YZM5110_Research_Project.pdf" "$DEST/" 2>/dev/null || true
date > "$DEST/LAST_BACKUP.txt"
du -sh "$DEST"
