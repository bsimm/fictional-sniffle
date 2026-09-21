#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "musicbrainzngs",
#   "librosa",
#   "numpy",
# ]
# ///
"""
mp3sd.py — Format SD card as FAT32 and populate with AAC 320 kbps audio.

Commands:
    scan   Walk a source directory and probe lossless audio files into the cache.
           Supports incremental chunked scanning so you can build the cache over
           multiple short runs without waiting for a full library probe up front.

    flash  Read the cache, optionally enrich genres via MusicBrainz, select
           tracks, convert to AAC 320 kbps, and write to a FAT32 SD card.

Usage:
    ./mp3sd.py scan  --source <dir> [<dir> ...] [--chunk N] [--rescan]
    ./mp3sd.py flash --disk <dev>   [--genre GENRE ...] [options]
"""

import argparse
import json
import os
import plistlib
import re
import stat
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from random import Random

import musicbrainzngs

# ── Configuration ────────────────────────────────────────────────────────────

# Extensions to scan; .m4a is included to catch ALAC (codec is verified below)
LOSSLESS_EXTENSIONS = {".flac", ".wav", ".aiff", ".aif", ".ape", ".wv", ".m4a"}
LOSSLESS_CODECS     = {
    "flac", "alac",
    "pcm_s16le", "pcm_s24le", "pcm_s32le",
    "pcm_s16be", "pcm_s24be", "pcm_s32be",
    "pcm_f32le", "pcm_f64le",
    "ape", "wavpack",
}

OUTPUT_BITRATE  = "320k"           # Tacoma maximum: 320 kbps
OUTPUT_EXT      = ".m4a"          # AAC-LC in MPEG-4 container (.m4a)
DEFAULT_LIMIT   = 9000            # stay safely under the 9999-track limit
BYTES_PER_SEC   = 320_000 // 8   # estimated output bytes/s at 320 kbps (40 KB/s)
MAX_BYTES       = 30 * 1024 ** 3  # 30 GiB ceiling (leaves FAT32 overhead on 32 GB card)
CACHE_FILE      = Path(__file__).parent / "scan_cache.json"
STAGING_DIR     = Path(__file__).parent / "converted"

MB_APP          = ("mp3sd", "1.0", "https://github.com/user/mp3sd")

# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list[str], check=True, capture=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def fat32_safe(name: str, max_len: int = 60) -> str:
    """Return a FAT32-safe version of a string (no illegal chars, ASCII-ish)."""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"[. _]{2,}", " ", name).strip(". ")
    return name[:max_len] or "Unknown"


def get_lossless_info(path: str) -> dict | None:
    """Use ffprobe to verify the file is lossless and extract metadata."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            path,
        ],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    audio_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"),
        None,
    )
    if audio_stream is None:
        return None

    codec = audio_stream.get("codec_name", "").lower()
    if codec not in LOSSLESS_CODECS:
        return None  # not lossless (e.g. an AAC .m4a)

    try:
        duration = float(
            audio_stream.get("duration") or data.get("format", {}).get("duration", 0)
        )
    except (TypeError, ValueError):
        duration = 0.0

    tags = data.get("format", {}).get("tags", {})
    tags = {k.lower(): v for k, v in tags.items()}

    return {
        "codec":           codec,
        "duration":        duration,
        "artist":          tags.get("album_artist") or tags.get("artist") or "",
        "album":           tags.get("album") or "",
        "title":           tags.get("title") or "",
        "track":           tags.get("track") or "",
        # MusicBrainz IDs — present if tagged by Picard / beets / etc.
        "mbid_recording":  tags.get("musicbrainz_trackid") or tags.get("musicbrainz_recordingid") or "",
        "mbid_release":    tags.get("musicbrainz_albumid") or "",
    }


def convert_to_aac(src: str, dest: Path) -> tuple[bool, str]:
    """Convert a lossless file to AAC 320 kbps .m4a, preserving metadata tags.
    Returns (success, error_message)."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", src,
            "-map", "0:a",
            "-map", "0:v?",
            "-c:v", "copy",
            "-disposition:v:0", "attached_pic",
            "-af", "aformat=sample_fmts=fltp:channel_layouts=stereo,aresample=44100",
            "-c:a", "aac", "-b:a", OUTPUT_BITRATE,
            "-map_metadata", "0",
            "-movflags", "+faststart",
            "-threads", "0",
            "-y",
            str(dest),
        ],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return True, ""
    # Return the last few non-empty stderr lines for context
    lines = [l.strip() for l in result.stderr.splitlines() if l.strip()]
    return False, " | ".join(lines[-3:]) if lines else "unknown error"


def parse_track_num(raw: str) -> str:
    """Return zero-padded 2-digit track number from a tag like '3' or '3/12'."""
    try:
        n = int(raw.split("/")[0])
        return f"{n:02d}"
    except (ValueError, AttributeError):
        return "00"


def derive_dest_path(src_path: str, info: dict) -> tuple[str, str, str]:
    """
    Return (artist_dir, album_dir, filename) for the destination, derived from
    tags where available, falling back to the source directory structure.
    """
    src = Path(src_path)
    parent_name      = src.parent.name
    grandparent_name = src.parent.parent.name

    artist_raw = info["artist"].strip()
    if not artist_raw:
        artist_raw = re.split(r"\s+[-–]\s+", grandparent_name)[0]
    if not artist_raw:
        artist_raw = grandparent_name or "Unknown Artist"
    artist_dir = fat32_safe(artist_raw)

    album_raw = info["album"].strip()
    if not album_raw:
        album_raw = parent_name or "Unknown Album"
    album_dir = fat32_safe(album_raw)

    title_raw = info["title"].strip()
    track_num = parse_track_num(info["track"])

    if title_raw:
        title = fat32_safe(title_raw)
    else:
        stem  = re.sub(r"^\d+[\s.\-_]+", "", src.stem)
        title = fat32_safe(stem) or fat32_safe(src.stem)

    filename = (
        f"{track_num} - {title}{OUTPUT_EXT}" if track_num != "00"
        else f"{title}{OUTPUT_EXT}"
    )

    return artist_dir, album_dir, filename


def deduplicate_filename(dest_file: Path) -> Path:
    """Append a numeric suffix if the destination file already exists."""
    if not dest_file.exists():
        return dest_file
    stem   = dest_file.stem
    suffix = dest_file.suffix
    parent = dest_file.parent
    i = 2
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


# ── Cache ─────────────────────────────────────────────────────────────────────
#
# Format: {"checked": [path, ...], "qualified": [{"path": ..., "info": ...}, ...]}
#
# "checked" is every path that has been probed (lossless or not), enabling
# incremental --chunk runs to skip already-processed files.
# Old flat-list caches ([{"path":..., "info":...}, ...]) are migrated on load.

def save_cache(qualified: list, checked: set) -> None:
    data = {
        "checked":   sorted(checked),
        "qualified": [{"path": path, "info": info} for path, info in qualified],
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[cache] {len(qualified)} qualified, {len(checked)} checked → {CACHE_FILE.name}")


def load_cache(force_rescan: bool) -> tuple[list, set]:
    """Return (qualified, checked_set). Both empty if missing/forced/corrupt."""
    if force_rescan or not CACHE_FILE.exists():
        return [], set()
    try:
        with open(CACHE_FILE) as f:
            raw = json.load(f)
    except json.JSONDecodeError:
        print("[cache] Cache corrupt; ignoring.")
        return [], set()

    # Migrate old flat-list format
    if isinstance(raw, list):
        print("[cache] Migrating old cache format ...")
        qualified = [(d["path"], d["info"]) for d in raw if os.path.exists(d["path"])]
        checked   = {path for path, _ in qualified}
        save_cache(qualified, checked)
        return qualified, checked

    try:
        checked   = set(raw.get("checked", []))
        qualified = [
            (d["path"], d["info"]) for d in raw.get("qualified", [])
            if os.path.exists(d["path"])
        ]
    except (KeyError, TypeError):
        print("[cache] Cache corrupt; ignoring.")
        return [], set()

    # Drop checked entries whose files no longer exist
    checked = {p for p in checked if os.path.exists(p)}

    dropped = len(raw.get("qualified", [])) - len(qualified)
    if dropped:
        print(f"[cache] Dropped {dropped} missing files.")

    return qualified, checked


# ── MusicBrainz ──────────────────────────────────────────────────────────────

def enrich_with_genres(qualified: list, checked: set) -> list:
    """
    Fetch genre tags from MusicBrainz for tracks that have MBID tags but no
    cached genre data yet.  Results are written back into each info dict and
    the cache is re-saved.  Tracks with no MBIDs are left with genres=[].
    """
    musicbrainzngs.set_useragent(*MB_APP)
    musicbrainzngs.set_rate_limit(True)   # honour 1 req/s without authentication

    needs_lookup = [
        (path, info) for path, info in qualified
        if "genres" not in info and (info.get("mbid_recording") or info.get("mbid_release"))
    ]

    # Mark tracks with no MBIDs so we don't re-attempt them on future runs
    for _, info in qualified:
        if "genres" not in info:
            info["genres"] = []

    if not needs_lookup:
        return qualified

    print(f"[mb] Fetching genre data for {len(needs_lookup)} tracks from MusicBrainz ...")
    enriched = 0

    for i, (path, info) in enumerate(needs_lookup, 1):
        if i % 10 == 0 or i == len(needs_lookup):
            print(f"  ... {i}/{len(needs_lookup)}", end="\r")

        genres: list[str] = []

        rec_id = info.get("mbid_recording", "")
        if rec_id:
            try:
                res    = musicbrainzngs.get_recording_by_id(rec_id, includes=["tags"])
                genres = [t["name"] for t in res["recording"].get("tag-list", [])]
            except musicbrainzngs.WebServiceError:
                pass

        if not genres:
            rel_id = info.get("mbid_release", "")
            if rel_id:
                try:
                    res    = musicbrainzngs.get_release_by_id(rel_id, includes=["tags"])
                    genres = [t["name"] for t in res["release"].get("tag-list", [])]
                except musicbrainzngs.WebServiceError:
                    pass

        info["genres"] = genres
        if genres:
            enriched += 1

    print(f"\n[mb] {enriched}/{len(needs_lookup)} tracks received genre data.")
    save_cache(qualified, checked)
    return qualified


def genre_matches(info: dict, genre_filter: set[str]) -> bool:
    return bool(genre_filter & {g.lower() for g in info.get("genres", [])})


# ── Mix ──────────────────────────────────────────────────────────────────────

# Camelot wheel — maps "<note> major/minor" to Camelot position e.g. "8B".
# Adjacent positions (±1 mod 12 same letter, or A↔B same number) blend well.
CAMELOT: dict[str, str] = {
    "C major":  "8B",  "A minor":  "8A",
    "G major":  "9B",  "E minor":  "9A",
    "D major": "10B",  "B minor": "10A",
    "A major": "11B",  "F# minor":"11A",
    "E major": "12B",  "C# minor":"12A",
    "B major":  "1B",  "G# minor": "1A",
    "F# major": "2B",  "D# minor": "2A",
    "C# major": "3B",  "A# minor": "3A",
    "G# major": "4B",  "F minor":  "4A",
    "D# major": "5B",  "C minor":  "5A",
    "A# major": "6B",  "G minor":  "6A",
    "F major":  "7B",  "D minor":  "7A",
}

# Krumhansl-Kessler tonal profiles for key detection via chroma correlation
_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NOTE_NAMES    = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CROSSFADE_SECS = 8    # default crossfade duration
ANALYSIS_OFFSET = 30   # seconds to skip before sampling
ANALYSIS_SECS   = 15   # seconds of audio loaded for BPM/key analysis


def analyze_track_audio(path: str) -> dict:
    """Detect BPM and musical key from audio using librosa (first ANALYSIS_SECS s)."""
    import librosa
    import numpy as np

    y, sr = librosa.load(path, mono=True, offset=ANALYSIS_OFFSET, duration=ANALYSIS_SECS)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    # Krumhansl-Kessler: correlate mean chroma against all 24 key profiles
    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    major       = np.array(_MAJOR_PROFILE)
    minor       = np.array(_MINOR_PROFILE)

    best_score, best_key = -np.inf, "C major"
    for i in range(12):
        for profile, mode in [(major, "major"), (minor, "minor")]:
            score = float(np.corrcoef(mean_chroma, np.roll(profile, i))[0, 1])
            if score > best_score:
                best_score = score
                best_key   = f"{_NOTE_NAMES[i]} {mode}"

    return {"bpm": round(bpm, 1), "key": best_key}


def enrich_with_analysis(qualified: list, checked: set) -> list:
    """Add 'bpm' and 'key' to each track's info via librosa. Updates cache."""
    needs = [(path, info) for path, info in qualified if "bpm" not in info]
    if not needs:
        return qualified

    print(f"[analysis] Analysing {len(needs)} tracks for BPM and key ...")
    for i, (path, info) in enumerate(needs, 1):
        name  = Path(path).name
        label = name if len(name) <= 50 else name[:47] + "..."
        print(f"\r[analysis] {i}/{len(needs)}  {label:<50}", end="", flush=True)
        try:
            info.update(analyze_track_audio(path))
        except Exception as exc:
            info["bpm"] = 0.0
            info["key"] = ""
            print(f"\n[analysis] {name}: {exc}")

    print(f"\r[analysis] Done.{' ' * 60}")
    save_cache(qualified, checked)
    return qualified


def camelot_distance(a: str, b: str) -> int:
    """
    Harmonic distance between two Camelot positions (e.g. "8B", "9A").
    0 = same key  1 = perfect blend  2 = workable  higher = avoid
    """
    if not a or not b:
        return 99
    num_a, letter_a = int(a[:-1]), a[-1]
    num_b, letter_b = int(b[:-1]), b[-1]
    num_dist = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
    if num_dist == 0:
        return 0 if letter_a == letter_b else 1   # relative major/minor
    if num_dist == 1 and letter_a == letter_b:
        return 1                                   # adjacent on the wheel
    return num_dist + (0 if letter_a == letter_b else 1)


def order_for_mix(tracks: list, rng: Random) -> list:
    """
    Greedy nearest-neighbour traversal of the Camelot wheel.
    Starts from a random track; each step picks the unused track with the
    lowest harmonic distance, breaking ties by closest BPM.
    """
    if not tracks:
        return tracks
    remaining = list(tracks)
    rng.shuffle(remaining)
    ordered = [remaining.pop(0)]
    while remaining:
        _, prev_info = ordered[-1]
        prev_cam = CAMELOT.get(prev_info.get("key", ""), "")
        prev_bpm = prev_info.get("bpm", 0.0)

        def score(item: tuple) -> tuple:
            _, info = item
            cam = CAMELOT.get(info.get("key", ""), "")
            return (camelot_distance(prev_cam, cam), abs(prev_bpm - info.get("bpm", 0.0)))

        best = min(remaining, key=score)
        remaining.remove(best)
        ordered.append(best)
    return ordered


def build_mix(ordered: list, output: Path, crossfade: int) -> None:
    """Concatenate tracks into a single mix using ffmpeg acrossfade."""
    n = len(ordered)
    if n == 0:
        print("[mix] No tracks.")
        return

    inputs = []
    for path, _ in ordered:
        inputs += ["-i", path]

    if n == 1:
        # Single track — just re-encode
        filter_complex = "[0:a]anull[mix]"
    else:
        # Chain acrossfade across all inputs:
        # [0:a][1:a]acrossfade=d=D[cf1]; [cf1][2:a]acrossfade=d=D[cf2]; ...
        parts = []
        prev  = "[0:a]"
        for i in range(1, n):
            out = f"[cf{i}]" if i < n - 1 else "[mix]"
            parts.append(f"{prev}[{i}:a]acrossfade=d={crossfade}:c1=tri:c2=tri{out}")
            prev = f"[cf{i}]"
        filter_complex = ";".join(parts)

    cmd = [
        "ffmpeg",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[mix]",
        "-c:a", "aac", "-b:a", OUTPUT_BITRATE,
        "-movflags", "+faststart",
        "-y",
        str(output),
    ]

    est_mins = (sum(i.get("duration", 0) for _, i in ordered) - (n - 1) * crossfade) / 60
    print(f"[mix] Encoding {n} tracks (~{est_mins:.0f} min) → {output} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        lines = [l.strip() for l in result.stderr.splitlines() if l.strip()]
        print(f"[error] ffmpeg: {lines[-1] if lines else 'unknown error'}")
    else:
        print("[mix] Done.")


# ── Disk operations ──────────────────────────────────────────────────────────

def validate_target_disk(disk: str) -> None:
    """Abort if disk fails safety checks; warn (with prompt) if not marked removable."""

    # Must be a block device
    try:
        if not stat.S_ISBLK(os.stat(disk).st_mode):
            print(f"[error] {disk} is not a block device.")
            sys.exit(1)
    except FileNotFoundError:
        print(f"[error] {disk} does not exist.")
        sys.exit(1)

    # Whole-disk nodes match /dev/diskN exactly — partitions have a trailing sN
    if not re.fullmatch(r"/dev/disk\d+", disk):
        print(f"[error] {disk} does not look like a whole-disk device (expected /dev/diskN).")
        sys.exit(1)

    result = subprocess.run(["diskutil", "info", "-plist", disk], capture_output=True)
    if result.returncode != 0:
        print(f"[error] diskutil info failed for {disk}: {result.stderr.decode().strip()}")
        sys.exit(1)

    info = plistlib.loads(result.stdout)

    if not info.get("WholeDisk", False):
        print(f"[error] {disk} is not a whole disk.")
        sys.exit(1)

    total_bytes = info.get("TotalSize", 0)
    if total_bytes >= 32 * 1024 ** 3:
        print(f"[error] {disk} is {total_bytes / 1024**3:.1f} GiB — this is probably not an SD card.")
        sys.exit(1)

    # Resolve the whole disk that contains the boot volume
    boot_result = subprocess.run(["diskutil", "info", "-plist", "/"], capture_output=True)
    if boot_result.returncode == 0:
        boot_info  = plistlib.loads(boot_result.stdout)
        boot_whole = boot_info.get("ParentWholeDisk", "")
        target_id  = info.get("DeviceIdentifier", "")
        if target_id and target_id == boot_whole:
            print(f"[error] {disk} is the boot disk — refusing to continue.")
            sys.exit(1)

    # Non-removable is a warning, not a hard failure (matches flash.py behaviour)
    if not info.get("RemovableMedia", False):
        print(f"[warn]  {disk} is not marked as removable media.")
        answer = input("  Continue anyway? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    print(f"[validate] {disk}: {total_bytes / 1024**3:.1f} GiB, not the boot disk. OK.")


def format_disk(disk: str, volume_name: str):
    """Erase and format disk as FAT32 (MBR) using diskutil on macOS."""
    disk_name = disk
    disk_gib  = ""
    result = subprocess.run(["diskutil", "info", "-plist", disk], capture_output=True)
    if result.returncode == 0:
        info      = plistlib.loads(result.stdout)
        disk_name = info.get("MediaName") or info.get("IORegistryEntryName") or disk
        total     = info.get("TotalSize", 0)
        disk_gib  = f"{total / 1024**3:.1f} GiB"

    print(f"\n{'─'*44}")
    print(f"  Target:  {disk}  ({disk_name}{', ' + disk_gib if disk_gib else ''})")
    print(f"  Volume:  {volume_name} (FAT32)")
    print(f"{'─'*44}")
    print(f"ALL DATA ON {disk} WILL BE ERASED.")
    print()
    confirm_disk = input("Re-type the disk path to confirm: ").strip()
    if confirm_disk != disk:
        print("Path mismatch. Aborting.")
        sys.exit(0)

    print(f"[format] Unmounting {disk} ...")
    run(["diskutil", "unmountDisk", disk])

    print(f"[format] Formatting {disk} as FAT32 ({volume_name}) ...")
    run(["diskutil", "eraseDisk", "MS-DOS", volume_name, "MBR", disk])
    print("[format] Done.\n")


def wait_for_mount(mount_point: str, retries: int = 10) -> bool:
    import time
    for _ in range(retries):
        if os.path.ismount(mount_point):
            return True
        time.sleep(1)
    return False


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_scan(args: argparse.Namespace) -> None:
    source_dirs = [os.path.expanduser(d) for d in args.source]
    for d in source_dirs:
        if not os.path.isdir(d):
            print(f"[error] Source directory not found: {d}")
            sys.exit(1)

    qualified, checked = load_cache(args.rescan)
    if qualified or checked:
        print(f"[cache] Resuming: {len(qualified)} qualified, {len(checked)} already checked.")

    # ── Walk for candidates ───────────────────────────────────────────────────
    scan_exts = (
        {f".{e.lstrip('.')}" .lower() for e in args.type}
        if args.type else LOSSLESS_EXTENSIONS
    )
    print(f"[scan] Walking {len(source_dirs)} director{'y' if len(source_dirs) == 1 else 'ies'} ...")
    candidates = []
    for source_dir in source_dirs:
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if f.startswith("._"):      # macOS AppleDouble resource fork
                    continue
                if Path(f).suffix.lower() in scan_exts:
                    candidates.append(os.path.join(root, f))
            print(f"\r[scan] {len(candidates)} candidates found ...", end="", flush=True)
    print(f"\r[scan] {len(candidates)} candidates found.          ")

    unchecked = [p for p in candidates if p not in checked]

    if not unchecked:
        print(f"[scan] All {len(candidates)} candidates already checked. "
              f"{len(qualified)} lossless in cache.")
        return

    to_probe = unchecked[:args.chunk] if args.chunk else unchecked
    remaining_after = len(unchecked) - len(to_probe)

    print(f"[scan] {len(unchecked)} unchecked — probing {len(to_probe)}"
          + (f", {remaining_after} remaining for next run." if remaining_after else "."))

    # ── Probe (parallel) ──────────────────────────────────────────────────────
    new_qualified = 0
    skipped       = 0
    errors        = 0
    done          = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(get_lossless_info, path): path for path in to_probe}
        for future in as_completed(futures):
            path  = futures[future]
            done += 1
            name  = Path(path).name
            label = name if len(name) <= 55 else name[:52] + "..."
            print(f"\r[scan] {done}/{len(to_probe)}  {label:<55}", end="", flush=True)

            checked.add(path)
            try:
                info = future.result()
            except subprocess.TimeoutExpired:
                errors += 1
                continue

            if info is None:
                skipped += 1
                continue

            qualified.append((path, info))
            new_qualified += 1

    print(f"\r[scan] +{new_qualified} lossless, {skipped} skipped, {errors} errors."
          + " " * 30)

    save_cache(qualified, checked)

    if remaining_after:
        print(f"[scan] {remaining_after} files not yet checked. Run scan again to continue.")


def cmd_flash(args: argparse.Namespace) -> None:
    qualified, checked = load_cache(False)
    if not qualified:
        print("[error] No scan cache found. Run:  ./mp3sd.py scan --source <dir> [<dir> ...]")
        sys.exit(1)

    print(f"[cache] {len(qualified)} qualified tracks loaded.")

    validate_target_disk(args.disk)

    rng = Random(args.seed)

    # ── Genre enrichment via MusicBrainz ─────────────────────────────────────
    genre_filter = {g.lower() for g in args.genre}
    if genre_filter and not args.no_mb:
        qualified = enrich_with_genres(qualified, checked)

    # ── Partition by genre, shuffle each pool, then fill ─────────────────────
    if genre_filter:
        genre_matched = [(p, i) for p, i in qualified if genre_matches(i, genre_filter)]
        others        = [(p, i) for p, i in qualified if not genre_matches(i, genre_filter)]
        rng.shuffle(genre_matched)
        rng.shuffle(others)
        print(f"[genre] {len(genre_matched)} tracks match {sorted(args.genre)}; "
              f"{len(others)} available as backfill.")
    else:
        genre_matched = qualified
        others        = []
        rng.shuffle(genre_matched)

    selected   = []
    cumulative = 0

    for pool in (genre_matched, others):
        for path, info in pool:
            if len(selected) >= args.limit:
                break
            est = int(info["duration"] * BYTES_PER_SEC)
            if cumulative + est > MAX_BYTES:
                break
            cumulative += est
            selected.append((path, info))

    genre_count = min(len(genre_matched), len(selected))
    fill_count  = len(selected) - genre_count

    if genre_filter:
        print(f"[select] {genre_count} genre-matched + {fill_count} backfill = {len(selected)} tracks "
              f"(~{cumulative / 1024**3:.2f} GiB estimated).")
    else:
        print(f"[select] {len(selected)} tracks (~{cumulative / 1024**3:.2f} GiB estimated).")

    # ── Build destination map ─────────────────────────────────────────────────
    dest_map: list[tuple[str, str, str, str]] = []
    for src_path, info in selected:
        artist_dir, album_dir, filename = derive_dest_path(src_path, info)
        dest_map.append((src_path, artist_dir, album_dir, filename))

    artists = len({a for _, a, _, _ in dest_map})
    albums  = len({(a, al) for _, a, al, _ in dest_map})
    print(f"[select] {albums} albums by {artists} artists.")

    if args.dry_run:
        print("[dry-run] Sample of planned layout:")
        for src, artist, album, fname in dest_map[:20]:
            print(f"  {artist}/{album}/{fname}")
        if len(dest_map) > 20:
            print(f"  ... and {len(dest_map)-20} more.")
        print("\n[dry-run] No disk operations performed.")
        return

    volume_name = args.volume.upper()
    mount_point = f"/Volumes/{volume_name}"

    # ── Format ────────────────────────────────────────────────────────────────
    if not args.no_format:
        format_disk(args.disk, volume_name)
    else:
        print("[format] Skipping format (--no-format).")

    if not wait_for_mount(mount_point):
        print(f"[error] {mount_point} did not appear. Is the card inserted?")
        sys.exit(1)

    print(f"[mount] Volume mounted at {mount_point}")

    # ── Convert locally (parallel), then copy to the card ────────────────────
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[convert] Encoding {len(dest_map)} tracks into {STAGING_DIR} ...")

    staged      = []  # (staged_path, artist, album, fname)
    converted   = 0
    copy_errors = 0

    def _convert_one(item):
        src, artist, album, fname = item
        stage_dir = STAGING_DIR / artist / album
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_file = deduplicate_filename(stage_dir / fname)
        ok, err = convert_to_aac(src, stage_file)
        return item, stage_file, ok, err

    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 4)) as pool:
        futures = {pool.submit(_convert_one, item): item for item in dest_map}
        for future in as_completed(futures):
            (src, artist, album, fname), stage_file, ok, err = future.result()
            if ok:
                converted += 1
                staged.append((stage_file, artist, album, fname))
                if converted % 10 == 0:
                    print(f"  [{converted}/{len(dest_map)}] converted ...", end="\r")
            else:
                print(f"\n[error] {Path(src).name}: {err}")
                copy_errors += 1

    print(f"\n[convert] Done. {converted} converted, {copy_errors} errors.")

    print(f"[copy] Copying {len(staged)} tracks to {mount_point} ...")
    import shutil
    copied = 0
    for stage_file, artist, album, fname in staged:
        dest_dir = Path(mount_point) / artist / album
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = deduplicate_filename(dest_dir / fname)
            shutil.copy2(stage_file, dest_file)
            copied += 1
            if copied % 10 == 0:
                print(f"  [{copied}/{len(staged)}] copied ...", end="\r")
        except OSError as e:
            print(f"\n[error] copy {stage_file} -> {dest_dir}: {e}")
            copy_errors += 1

    print(f"\n[done] Converted {converted} tracks, copied {copied}, {copy_errors} errors. "
          f"Staged files kept in {STAGING_DIR}")

    # ── Eject ─────────────────────────────────────────────────────────────────
    print(f"[eject] Ejecting {mount_point} ...")
    result = subprocess.run(["diskutil", "eject", mount_point], capture_output=True, text=True)
    if result.returncode == 0:
        print("[eject] Safe to remove the card.")
    else:
        print(f"[eject] Could not auto-eject: {result.stderr.strip()}")


def cmd_mix(args: argparse.Namespace) -> None:
    qualified, checked = load_cache(False)
    if not qualified:
        print("[error] No scan cache. Run:  ./mp3sd.py scan --source <dir> [<dir> ...]")
        sys.exit(1)

    print(f"[cache] {len(qualified)} tracks loaded.")

    genre_filter = {g.lower() for g in args.genre}
    if genre_filter and not args.no_mb:
        qualified = enrich_with_genres(qualified, checked)

    pool = (
        [(p, i) for p, i in qualified if genre_matches(i, genre_filter)]
        if genre_filter else list(qualified)
    )
    if genre_filter:
        print(f"[genre] {len(pool)} tracks match {sorted(args.genre)}.")

    if not pool:
        print("[mix] No tracks match the selected criteria.")
        sys.exit(0)

    pool = enrich_with_analysis(pool, checked)

    # Trim pool to fit target duration (estimated from raw track durations)
    if args.duration:
        h, m, s = (args.duration.split(":") + ["0", "0", "0"])[:3]
        target_secs   = int(h) * 3600 + int(m) * 60 + int(s)
        selected, acc = [], 0
        for path, info in pool:
            if acc - len(selected) * args.crossfade >= target_secs:
                break
            selected.append((path, info))
            acc += info.get("duration", 0)
        pool = selected

    rng     = Random(args.seed)
    ordered = order_for_mix(pool, rng)

    print(f"[mix] Order ({len(ordered)} tracks):")
    for path, info in ordered[:8]:
        key = info.get("key", "?")
        bpm = info.get("bpm", 0)
        cam = CAMELOT.get(key, "?")
        print(f"  {cam:>3}  {bpm:5.1f} bpm  {Path(path).stem[:55]}")
    if len(ordered) > 8:
        print(f"  ... and {len(ordered) - 8} more.")

    build_mix(ordered, Path(args.output).expanduser(), args.crossfade)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # ── scan ──────────────────────────────────────────────────────────────────
    p_scan = sub.add_parser("scan", help="Probe source directory into the cache.")
    p_scan.add_argument("--source",  required=True, nargs="+", metavar="DIR",
                        help="One or more directories to scan recursively for lossless audio.")
    p_scan.add_argument("--type",    nargs="+", metavar="EXT",
                        help="Limit scan to specific extensions, e.g. --type flac wav")
    p_scan.add_argument("--chunk",   type=int, default=0, metavar="N",
                        help="Probe at most N unchecked files per run (0 = all).")
    _default_workers = min(32, (os.cpu_count() or 4) + 4)
    p_scan.add_argument("--workers", type=int, default=_default_workers, metavar="N",
                        help=f"Parallel ffprobe workers (default: {_default_workers}).")
    p_scan.add_argument("--rescan",  action="store_true",
                        help="Clear cache and re-probe everything.")

    # ── flash ─────────────────────────────────────────────────────────────────
    p_flash = sub.add_parser("flash", help="Select, convert, and write tracks to SD card.")
    p_flash.add_argument("--disk",      required=True, metavar="DEV",
                         help="Target disk device, e.g. /dev/disk2.")
    p_flash.add_argument("--genre",     action="append", metavar="GENRE", default=[],
                         help="Genre filter (repeatable). Fills card with matched tracks first, "
                              "then backfills with random selections.")
    p_flash.add_argument("--volume",    default="MUSIC", metavar="NAME",
                         help="FAT32 volume label (default: MUSIC).")
    p_flash.add_argument("--dry-run",   action="store_true",
                         help="Preview layout without writing anything.")
    p_flash.add_argument("--no-format", action="store_true",
                         help="Skip the erase/format step.")
    p_flash.add_argument("--no-mb",     action="store_true",
                         help="Skip MusicBrainz genre lookup.")
    p_flash.add_argument("--limit",     type=int, default=DEFAULT_LIMIT, metavar="N",
                         help=f"Max tracks to copy (default: {DEFAULT_LIMIT}).")
    p_flash.add_argument("--seed",      type=int, default=42, metavar="N",
                         help="Random seed for reproducible shuffling.")

    # ── mix ───────────────────────────────────────────────────────────────────
    p_mix = sub.add_parser("mix", help="Build a DJ-style mix from cached tracks.")
    p_mix.add_argument("--output",    required=True, metavar="FILE",
                       help="Output file, e.g. ~/mix.m4a.")
    p_mix.add_argument("--genre",     action="append", metavar="GENRE", default=[],
                       help="Genre filter (repeatable).")
    p_mix.add_argument("--duration",  metavar="H:MM:SS",
                       help="Target mix duration, e.g. 1:30:00.")
    p_mix.add_argument("--crossfade", type=int, default=CROSSFADE_SECS, metavar="SECS",
                       help=f"Crossfade length in seconds (default: {CROSSFADE_SECS}).")
    p_mix.add_argument("--no-mb",     action="store_true",
                       help="Skip MusicBrainz genre lookup.")
    p_mix.add_argument("--seed",      type=int, default=42, metavar="N",
                       help="Random seed.")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "mix":
        cmd_mix(args)
    else:
        cmd_flash(args)


if __name__ == "__main__":
    main()
