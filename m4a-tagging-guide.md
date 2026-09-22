# Fixing tags on M4A files — process and commands

Reusable notes from tagging a directory tree of bootleg concert `.m4a` files
(ALAC-in-MP4) that only had `©ART` (artist) and `©alb` (album) set, with no
title, track/disc number, year, or cover art.

## 1. Set up tooling

```bash
which ffprobe ffmpeg          # sanity check, not strictly required
python3 -c "import mutagen"   # check if already installed
pip3 install --user --break-system-packages mutagen
```

`mutagen` ships a CLI inspector at
`~/Library/Python/<ver>/bin/mutagen-inspect` (add that dir to `PATH`, or call
it by full path).

## 2. Find the files and see what's there

```bash
find "<root>" -iname "*.m4a" | sort

# Inspect one file per subdirectory to see the current tag state
cd "<root>"
for d in */; do
  f=$(find "$d" -iname "*.m4a" | head -1)
  echo "=== $d ==="
  mutagen-inspect "$f"
done
```

Look at which atoms are present/missing: `©nam` (title), `©ART` (artist),
`©alb` (album), `©day` (year), `trkn` (track/total), `disk` (disc/total),
`covr` (cover art).

## 3. Establish ground truth for the missing data

File names usually only encode disc/track numbers (e.g. `d1t01`, `t09`), not
song titles. Don't invent titles — track them down:

- Check the directory itself for sidecar `.txt`/`.cue`/`.nfo` files with a
  tracklist.
- If the source is an archive.org item, check its metadata for text files:
  `curl -s "https://archive.org/metadata/<identifier>"`.
- Otherwise, search the web for the artist + exact date (setlist.fm, band's
  own live-show archive, taper community listings). Match the found setlist's
  song count against each disc's track count before trusting it — mismatches
  usually mean intro/banter/tuning is either split into its own track or
  merged into a song, or the found setlist is for the wrong show.
- If a segment genuinely can't be identified (crowd noise, unlabeled bit),
  tag it honestly (e.g. `"Song #1"`) rather than guessing a real title.

## 4. Write tags with mutagen (Python)

`mutagen-inspect` is read-only; use a small Python script with `mutagen.mp4`
to write tags. MP4 atom keys:

| Atom   | Meaning              |
|--------|----------------------|
| `©nam` | title                |
| `©ART` | artist               |
| `©alb` | album                |
| `©day` | year (string)        |
| `trkn` | `[(track, total)]`   |
| `disk` | `[(disc, total)]`    |
| `covr` | `[MP4Cover(bytes)]`  |

```python
#!/usr/bin/env python3
from mutagen.mp4 import MP4, MP4Cover
from pathlib import Path

# One entry per file: (path, title, disc, disc_total, track, track_total)
PLAN = [
    ("fugazi1989-10-08.flac16/fuqazi1989-10-08d1t01.m4a", "Intro", None, None, 1, 21),
    # ...
]

ARTIST = "Fugazi"
ALBUM = "1989-10-08 - Maxwell's, Hoboken, NJ"
YEAR = "1989"
COVER_PATH = "cover-fugazi1989-10-08.jpg"   # optional, JPEG

cover_data = Path(COVER_PATH).read_bytes() if COVER_PATH else None

for rel, title, disc, disc_total, track, track_total in PLAN:
    f = MP4(rel)
    f["©nam"] = [title]
    f["©ART"] = [ARTIST]
    f["©alb"] = [ALBUM]
    f["©day"] = [YEAR]
    f["trkn"] = [(track, track_total)]
    if disc is not None:
        f["disk"] = [(disc, disc_total)]
    if cover_data:
        f["covr"] = [MP4Cover(cover_data, imageformat=MP4Cover.FORMAT_JPEG)]
    f.save()
```

Run once per show/subdirectory (different album/year/cover per folder), or
generalize the script to loop over subdirectories and derive `disc`/`track`
from the filename pattern (e.g. regex `d(\d+)t(\d+)` or `t(\d+)`).

## 5. Verify

```bash
for d in */; do
  echo "=== $d ==="
  mutagen-inspect "$d"/*.m4a | grep -E '^--|©nam|©alb|©ART|©day|trkn|disk|covr'
done
```

Spot-check that track counts match `trkn` totals, disc totals match the
number of discs in that folder, and titles read as real song names (not
leftover filename fragments).

## 6. Sourcing cover art from archive.org

Archive.org auto-generates a tiny `__ia_thumb.jpg` (~2–3 KB) for every item —
not usable as art. Look instead for a larger uploaded image file in the same
metadata listing (often the taper's scanned flyer/ticket/cover):

```bash
curl -s "https://archive.org/metadata/<identifier>" | python3 -c '
import json, sys
d = json.load(sys.stdin)
for f in d.get("files", []):
    n = f.get("name", "")
    if n.lower().endswith((".jpg", ".jpeg", ".png")) and "spectrogram" not in n.lower():
        print(n, f.get("size"))
'

curl -sL "https://archive.org/download/<identifier>/<file>.jpg" -o cover.jpg
```

Prefer a file that's clearly larger than the `__ia_thumb.jpg` and not named
`*_spectrogram*` (those are per-track waveform/frequency plots, not artwork).
