# mp3sd

Prepares a 32 GB SD card for the Tacoma head unit. Scans a source directory for
lossless audio files, converts them to AAC 320 kbps `.m4a` (the highest quality
format/bitrate the head unit supports), and writes everything to a FAT32-formatted
card. Optionally filters by genre using MusicBrainz metadata.

## Requirements

- macOS (uses `diskutil`)
- [`uv`](https://github.com/astral-sh/uv) — manages the Python environment and dependencies
- `ffmpeg` / `ffprobe` — audio probing and conversion (`brew install ffmpeg`)

## Usage

Scanning and flashing are separate commands. Build the cache with `scan` first,
then `flash` whenever you want to write a card.

```sh
# Probe one or more directories into the cache (parallelised)
./mp3sd.py scan --source ~/Music
./mp3sd.py scan --source ~/Music ~/ExternalDrive/Music

# Scan only FLAC files
./mp3sd.py scan --source ~/Music --type flac

# Scan specific types
./mp3sd.py scan --source ~/Music --type flac wav aiff

# Scan in chunks — useful for large libraries; resumes where it left off
./mp3sd.py scan --source ~/Music --chunk 50

# Flash the card using whatever is in the cache
./mp3sd.py flash --disk /dev/disk2

# Genre filter — fill with jazz and blues first, backfill remaining space randomly
./mp3sd.py flash --disk /dev/disk2 --genre jazz --genre blues

# Dry run — preview the layout without touching the card
./mp3sd.py flash --disk /dev/disk2 --dry-run

# Skip reformatting (card is already FAT32)
./mp3sd.py flash --disk /dev/disk2 --no-format

# Re-probe everything from scratch
./mp3sd.py scan --source ~/Music --rescan

# Genre filter without hitting the MusicBrainz API (use cached genres only)
./mp3sd.py flash --disk /dev/disk2 --genre jazz --no-mb

# Build a 90-minute DJ mix, harmonically ordered
./mp3sd.py mix --output ~/mix.m4a --duration 1:30:00

# Genre-filtered mix with a longer crossfade
./mp3sd.py mix --output ~/jazz-mix.m4a --genre jazz --crossfade 12
```

## Options

### `scan`

| Flag | Default | Description |
|---|---|---|
| `--source DIR [DIR ...]` | required | One or more directories to scan recursively for lossless audio |
| `--type EXT [EXT ...]` | all lossless | Limit scan to specific extensions, e.g. `flac` or `flac wav aiff` |
| `--chunk N` | 0 (all) | Probe at most N unchecked files per run; run again to continue |
| `--workers N` | `cpu_count + 4` | Parallel ffprobe workers |
| `--rescan` | off | Clear cache and re-probe everything |

### `flash`

| Flag | Default | Description |
|---|---|---|
| `--disk DEV` | required | Target disk, e.g. `/dev/disk2` |
| `--genre GENRE` | — | Genre filter (repeatable). Matched tracks fill the card first; remaining space is backfilled randomly. Requires MusicBrainz MBID tags. |
| `--volume NAME` | `MUSIC` | FAT32 volume label |
| `--limit N` | `9000` | Max tracks (head unit limit is 9999) |
| `--seed N` | `42` | Random seed for reproducible shuffling |
| `--dry-run` | off | Preview layout without writing anything |
| `--no-format` | off | Skip the erase/format step |
| `--no-mb` | off | Skip MusicBrainz lookup; use cached genre data only |

### `mix`

| Flag | Default | Description |
|---|---|---|
| `--output FILE` | required | Output path, e.g. `~/mix.m4a` |
| `--duration H:MM:SS` | — | Target mix length; trims track pool to fit |
| `--crossfade SECS` | `8` | Crossfade length in seconds |
| `--genre GENRE` | — | Genre filter (repeatable) |
| `--no-mb` | off | Skip MusicBrainz genre lookup |
| `--seed N` | `42` | Random seed |

## How it works

### `scan` / `flash`

1. **Scan** — walks `--source` for `.flac`, `.wav`, `.aiff`, `.aif`, `.ape`, `.wv`, and `.m4a` files. Probes each with `ffprobe` in parallel to verify it is actually lossless (FLAC, ALAC, PCM, APE, WavPack). Results are cached in `scan_cache.json` alongside a `checked` set so incremental `--chunk` runs skip already-probed files.
2. **Genre enrichment** — if `--genre` is given, looks up genre tags on MusicBrainz for tracks that have `MUSICBRAINZ_TRACKID` or `MUSICBRAINZ_ALBUMID` tags. Results are cached; only tracks missing genre data hit the API. No authentication required.
3. **Select** — shuffles the library (seeded), applies the genre priority fill, and trims to the 30 GiB / 9000-track limits.
4. **Validate** — checks that `--disk` is a block device, a whole disk (not a partition), under 32 GiB, and not the boot disk. Prompts to continue if not marked removable.
5. **Format** — runs `diskutil eraseDisk MS-DOS <VOLUME> MBR <disk>` after requiring the disk path to be re-typed as confirmation.
6. **Convert** — runs `ffmpeg` to encode each track to AAC 320 kbps `.m4a`, preserving metadata tags. Files are laid out as `Artist/Album/NN - Title.m4a`.
7. **Eject** — calls `diskutil eject` when done.

### `mix`

1. **Analysis** — detects BPM and musical key for each track using librosa. Samples a 15-second window at 00:30–00:45 (skips intros, fast enough for large libraries). Results are stored in `scan_cache.json` so re-runs skip already-analysed tracks.
2. **Key detection** — uses Krumhansl-Kessler tonal profiles correlated against chroma features to identify the most likely key from all 24 major/minor candidates.
3. **Harmonic ordering** — maps each key to its [Camelot wheel](https://mixedinkey.com/camelot-wheel/) position and sorts tracks using a greedy nearest-neighbour traversal (lowest harmonic distance, BPM proximity as tiebreaker).
4. **Mix assembly** — builds an ffmpeg `filter_complex` chaining `acrossfade=c1=tri:c2=tri` across all inputs and encodes the result to AAC 320 kbps `.m4a`.

## Genre filtering notes

- Genre data comes from MusicBrainz crowd-sourced tags and requires that your files are tagged with MusicBrainz IDs (e.g. by [MusicBrainz Picard](https://picard.musicbrainz.org/) or [beets](https://beets.io/)).
- Tracks without MBIDs are silently excluded from genre matching and go into the backfill pool.
- The MusicBrainz API allows 1 request/second without authentication. Large unenriched libraries will take time on first run; subsequent runs use the cache.
- `--genre` is case-insensitive and matches any of the supplied values (OR logic).

---

# From Owner's Manual
## USB memory

### Compatible devices

- USB memory device that can be used for MP3, WMA and AAC playback.

### Compatible device formats

- USB communication format: USB2.0 HS (480 Mbps) and FS (12 Mbps)
- File system format: FAT16/32 (Windows)
- Correspondence class: Mass storage class
- MP3, WMA and AAC files written to a device with any format other than those listed above may not play correctly, and their file names and folder names may not be displayed correctly.

Standards and limitations:

- Maximum directory hierarchy: 8 levels
- Maximum number of folders in a device: 3000 (including the root)
- Maximum number of files in a device: 9999
- Maximum number of files per folder: 255

### MP3, WMA and AAC files

- MP3 (MPEG Audio LAYER 3) is a standard audio compression format. Files can be compressed to approximately 1/10 of their original size using MP3 compression. WMA (Windows Media Audio) is a Microsoft audio compression format. This format compresses audio data to a size smaller than that of the MP3 format.
- AAC is short for Advanced Audio Coding and refers to an audio compression technology standard used with MPEG2 and MPEG4.
- MP3, WMA and AAC file and media/formats compatibility are limited.

### MP3 file compatibility

- Compatible standards: MP3 (MPEG1 AUDIO LAYER II, III / MPEG2 AUDIO LAYER II, III)
- Compatible sampling frequencies
  - MPEG1 AUDIO LAYER II, III: 32, 44.1, 48 kHz
  - MPEG2 AUDIO LAYER II, III: 16, 22.05, 24 kHz
- Compatible bit rates (VBR supported)
  - MPEG1 AUDIO LAYER II, III: 32–320 kbps
  - MPEG2 AUDIO LAYER II, III: 8–160 kbps
- Compatible channel modes: stereo, joint stereo, dual channel, mono

### WMA file compatibility

- Compatible standards: WMA Ver. 7, 8, 9
- Compatible sampling frequencies (High Profile): 32, 44.1, 48 kHz
- Compatible bit rates (High Profile): 48–320 kbps (VBR)

### AAC file compatibility

- Compatible standards: MPEG4/AAC-LC
- Compatible sampling frequencies: 11.025, 12, 16, 22.05, 24, 32, 44.1, 48 kHz
- Compatible bit rates: 16–320 kbps
- Compatible channel modes: 1 ch, 2 ch

### File names

- The only files that can be recognized as MP3/WMA/AAC and played are those with the extension `.mp3`, `.wma`, or `.m4a`.
- If the extensions `.mp3`, `.wma`, or `.m4a` are used for files other than MP3, WMA and AAC files, they will be skipped (not played).

### ID3, WMA and AAC tags

- ID3 tags can be added to MP3 files, making it possible to record the track title, artist name, etc. Compatible with ID3 Ver. 1.0, 1.1, 2.2, and 2.3. (Character count is based on Ver. 1.0 and 1.1.)
- WMA tags can be added to WMA files, making it possible to record the track title and artist name.
- AAC tags can be added to AAC files, making it possible to record the track title and artist name.

### Playback

- When a USB memory device is connected, all files are checked before playback begins. To make the file check finish more quickly, do not include any files other than MP3, WMA and AAC files or create any unnecessary folders.
- When the audio source is changed to USB memory mode, the device starts playing the first file in the first folder. If the same device is removed and reconnected (contents unchanged), playback resumes from where it left off.
