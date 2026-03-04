#!/usr/bin/env python3
"""
mp3sd.py — Format SD card as FAT32 and populate with 320kbps MP3s.

Usage:
    python3 mp3sd.py --source <dir> --disk <dev> [options]

Required:
    --source DIR  Directory to scan for MP3s (searched recursively).
    --disk DEV    Raw disk device to format, e.g. /dev/disk2 (macOS).

Options:
    --dry-run     Scan and report without formatting or copying anything.
    --no-format   Skip disk erasure/format step (card already FAT32).
    --volume NAME Volume label for the FAT32 partition (default: MUSIC).
    --limit N     Max tracks to copy (default: 9000, Tacoma limit is 9999).
    --seed N      Random seed for reproducible shuffling.
    --rescan      Ignore cache and re-probe all MP3s.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from random import Random

# ── Configuration ────────────────────────────────────────────────────────────

TARGET_BITRATE_KBPS = 320           # only copy files at this bitrate
BITRATE_TOLERANCE   = 8             # ± kbps tolerance (handles CBR headers)
DEFAULT_LIMIT       = 9000          # stay safely under the 9999-track limit
MAX_BYTES           = 30 * 1024 ** 3  # 30 GiB ceiling (leaves room for FAT32 overhead on 32GB card)
CACHE_FILE          = Path(__file__).parent / "scan_cache.json"

# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list[str], check=True, capture=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def fat32_safe(name: str, max_len: int = 60) -> str:
    """Return a FAT32-safe version of a string (no illegal chars, ASCII-ish)."""
    # Normalize unicode → closest ASCII equivalent
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    # Replace FAT32-illegal chars with underscore
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    # Collapse runs of spaces/dots/underscores
    name = re.sub(r"[. _]{2,}", " ", name).strip(". ")
    return name[:max_len] or "Unknown"


def get_audio_info(path: str) -> dict | None:
    """Use ffprobe to get bitrate and ID3 tags from an mp3."""
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

    bit_rate_raw = audio_stream.get("bit_rate") or data.get("format", {}).get("bit_rate")
    try:
        bit_rate_kbps = int(bit_rate_raw) // 1000
    except (TypeError, ValueError):
        bit_rate_kbps = 0

    tags = data.get("format", {}).get("tags", {})
    # ID3 tag keys are case-insensitive; normalize
    tags = {k.lower(): v for k, v in tags.items()}

    return {
        "bitrate": bit_rate_kbps,
        "artist":  tags.get("album_artist") or tags.get("artist") or "",
        "album":   tags.get("album") or "",
        "title":   tags.get("title") or "",
        "track":   tags.get("track") or "",
    }


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
    ID3 tags where available, falling back to the source directory structure.
    """
    src = Path(src_path)
    parent_name = src.parent.name          # immediate album folder name
    grandparent_name = src.parent.parent.name  # artist or collection folder

    # ── Artist ──────────────────────────────────────────────────────────────
    artist_raw = info["artist"].strip()
    if not artist_raw:
        # Heuristic: "Artist - Album (year)" or "Artist"
        artist_raw = re.split(r"\s+[-–]\s+", grandparent_name)[0]
    if not artist_raw:
        artist_raw = grandparent_name or "Unknown Artist"
    artist_dir = fat32_safe(artist_raw)

    # ── Album ────────────────────────────────────────────────────────────────
    album_raw = info["album"].strip()
    if not album_raw:
        album_raw = parent_name or "Unknown Album"
    album_dir = fat32_safe(album_raw)

    # ── Filename ─────────────────────────────────────────────────────────────
    title_raw = info["title"].strip()
    track_num = parse_track_num(info["track"])

    if title_raw:
        title = fat32_safe(title_raw)
    else:
        # Strip leading track numbers from filename
        stem = re.sub(r"^\d+[\s.\-_]+", "", src.stem)
        title = fat32_safe(stem) or fat32_safe(src.stem)

    filename = f"{track_num} - {title}.mp3" if track_num != "00" else f"{title}.mp3"

    return artist_dir, album_dir, filename


def deduplicate_filename(dest_file: Path) -> Path:
    """Append a numeric suffix if the destination file already exists."""
    if not dest_file.exists():
        return dest_file
    stem = dest_file.stem
    suffix = dest_file.suffix
    parent = dest_file.parent
    i = 2
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


# ── Cache ────────────────────────────────────────────────────────────────────

def save_cache(qualified: list) -> None:
    """Persist [(path, info), ...] to JSON so future runs skip ffprobe."""
    data = [{"path": path, "info": info} for path, info in qualified]
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[cache] Saved {len(data)} entries to {CACHE_FILE}")


def load_cache(force_rescan: bool) -> list | None:
    """Return cached list or None if missing/stale/forced."""
    if force_rescan or not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        # Drop entries whose source file no longer exists
        valid = [(d["path"], d["info"]) for d in data if os.path.exists(d["path"])]
        if len(valid) < len(data):
            print(f"[cache] Dropped {len(data) - len(valid)} missing files from cache.")
        return valid
    except (json.JSONDecodeError, KeyError):
        print("[cache] Cache corrupt; ignoring.")
        return None


# ── Disk operations ──────────────────────────────────────────────────────────

def confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [yes/no]: ").strip().lower()
    return answer in ("yes", "y")


def format_disk(disk: str, volume_name: str):
    """Erase and format disk as FAT32 (MBR) using diskutil on macOS."""
    print(f"\n{'='*60}")
    print(f"  WARNING: This will ERASE ALL DATA on {disk}")
    print(f"  Volume name: {volume_name}")
    print(f"{'='*60}\n")
    if not confirm(f"Erase {disk} and format as FAT32?"):
        print("Aborted by user.")
        sys.exit(0)

    print(f"[format] Unmounting {disk} ...")
    run(["diskutil", "unmountDisk", disk])

    print(f"[format] Formatting {disk} as FAT32 ({volume_name}) ...")
    # eraseDisk MS-DOS uses FAT32 for disks >= 2GB; MBR for broad compatibility
    run(["diskutil", "eraseDisk", "MS-DOS", volume_name, "MBR", disk])
    print("[format] Done.\n")


def wait_for_mount(mount_point: str, retries: int = 10) -> bool:
    import time
    for _ in range(retries):
        if os.path.ismount(mount_point):
            return True
        time.sleep(1)
    return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source",    required=True, metavar="DIR",  help="Source directory to scan for MP3s.")
    parser.add_argument("--disk",      required=True, metavar="DEV",  help="Target disk device, e.g. /dev/disk2.")
    parser.add_argument("--volume",    default="MUSIC", metavar="NAME", help="FAT32 volume label (default: MUSIC).")
    parser.add_argument("--dry-run",   action="store_true", help="Scan only; no disk writes.")
    parser.add_argument("--no-format", action="store_true", help="Skip format step.")
    parser.add_argument("--limit",     type=int, default=DEFAULT_LIMIT, metavar="N")
    parser.add_argument("--seed",      type=int, default=42, metavar="N")
    parser.add_argument("--rescan",    action="store_true", help="Ignore cache and re-probe all MP3s.")
    args = parser.parse_args()

    source_dir  = os.path.expanduser(args.source)
    target_disk = args.disk
    volume_name = args.volume.upper()
    mount_point = f"/Volumes/{volume_name}"

    if not os.path.isdir(source_dir):
        print(f"[error] Source directory not found: {source_dir}")
        sys.exit(1)

    rng = Random(args.seed)

    # ── 1. Scan source for 320kbps MP3s (with cache) ────────────────────────
    qualified = load_cache(args.rescan)
    if qualified is None:
        print(f"[scan] Walking {source_dir} for MP3s ...")
        all_mp3s = []
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if f.lower().endswith(".mp3"):
                    all_mp3s.append(os.path.join(root, f))

        print(f"[scan] Found {len(all_mp3s)} .mp3 files. Checking bitrates ...")

        qualified = []
        skipped   = 0
        errors    = 0

        lo = TARGET_BITRATE_KBPS - BITRATE_TOLERANCE
        hi = TARGET_BITRATE_KBPS + BITRATE_TOLERANCE

        for i, path in enumerate(all_mp3s, 1):
            if i % 100 == 0:
                print(f"  ... {i}/{len(all_mp3s)} checked, {len(qualified)} qualify so far", end="\r")
            try:
                info = get_audio_info(path)
            except subprocess.TimeoutExpired:
                errors += 1
                continue

            if info is None:
                errors += 1
                continue

            if lo <= info["bitrate"] <= hi:
                qualified.append((path, info))
            else:
                skipped += 1

        print(f"\n[scan] {len(qualified)} qualifying 320kbps tracks, "
              f"{skipped} skipped (wrong bitrate), {errors} errors.")
        save_cache(qualified)
    else:
        print(f"[scan] Loaded {len(qualified)} tracks from cache ({CACHE_FILE.name}). "
              f"Use --rescan to rebuild.")

    if not qualified:
        print("[scan] Nothing to copy. Exiting.")
        sys.exit(0)

    # ── 2. Shuffle, then trim to track limit and 30 GiB size cap ────────────
    rng.shuffle(qualified)
    if len(qualified) > args.limit:
        print(f"[select] Trimming to {args.limit} random tracks (seed={args.seed}).")
        qualified = qualified[: args.limit]

    # Enforce size cap: walk the shuffled list and cut once we'd exceed MAX_BYTES
    cumulative = 0
    size_trimmed = []
    for item in qualified:
        size = os.path.getsize(item[0])
        if cumulative + size > MAX_BYTES:
            break
        cumulative += size
        size_trimmed.append(item)
    if len(size_trimmed) < len(qualified):
        print(f"[select] Size cap hit at {cumulative / 1024**3:.1f} GiB — "
              f"dropping {len(qualified) - len(size_trimmed)} tracks.")
    qualified = size_trimmed

    # ── 3. Build destination map ──────────────────────────────────────────────
    # Map src_path → (artist_dir, album_dir, filename)
    dest_map: list[tuple[str, str, str, str]] = []
    for src_path, info in qualified:
        artist_dir, album_dir, filename = derive_dest_path(src_path, info)
        dest_map.append((src_path, artist_dir, album_dir, filename))

    # Stats
    artists = len({a for _, a, _, _ in dest_map})
    albums  = len({(a, al) for _, a, al, _ in dest_map})
    print(f"[select] {len(dest_map)} tracks across {albums} albums by {artists} artists.")

    if args.dry_run:
        total_size = sum(os.path.getsize(s) for s, *_ in dest_map)
        print(f"\n[dry-run] Total size: {total_size / 1024**3:.2f} GiB "
              f"(cap: {MAX_BYTES / 1024**3:.0f} GiB)")
        print("[dry-run] Sample of planned layout:")
        for src, artist, album, fname in dest_map[:20]:
            print(f"  {artist}/{album}/{fname}")
        if len(dest_map) > 20:
            print(f"  ... and {len(dest_map)-20} more.")
        print("\n[dry-run] No disk operations performed.")
        return

    # ── 4. Format disk ────────────────────────────────────────────────────────
    if not args.no_format:
        format_disk(target_disk, volume_name)
    else:
        print("[format] Skipping format (--no-format).")

    if not wait_for_mount(mount_point):
        print(f"[error] {mount_point} did not appear after formatting. Is the card inserted?")
        sys.exit(1)

    print(f"[mount] Volume mounted at {mount_point}")

    # ── 5. Copy files ─────────────────────────────────────────────────────────
    copied  = 0
    copy_errors = 0

    for src, artist, album, fname in dest_map:
        dest_dir = Path(mount_point) / artist / album
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"\n[error] mkdir {dest_dir}: {e}")
            copy_errors += 1
            continue

        dest_file = deduplicate_filename(dest_dir / fname)
        try:
            shutil.copy2(src, dest_file)
            copied += 1
            if copied % 50 == 0:
                print(f"  [{copied}/{len(dest_map)}] copied ...", end="\r")
        except OSError as e:
            print(f"\n[error] copy {src} → {dest_file}: {e}")
            copy_errors += 1

    print(f"\n[done] Copied {copied} tracks, {copy_errors} errors.")

    # ── 6. Eject ──────────────────────────────────────────────────────────────
    print(f"[eject] Ejecting {mount_point} ...")
    result = subprocess.run(["diskutil", "eject", mount_point], capture_output=True, text=True)
    if result.returncode == 0:
        print("[eject] Safe to remove the card.")
    else:
        print(f"[eject] Could not auto-eject: {result.stderr.strip()}")


if __name__ == "__main__":
    main()
