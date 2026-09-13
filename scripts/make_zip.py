"""Package VoiceGuard project into a ZIP for submission."""

import zipfile
import os
import hashlib

ROOT = "D:/APPs/Intel苏州线下比赛/voiceguard"
OUT = "D:/APPs/Intel苏州线下比赛/voiceguard_submission_v6.zip"
EXCLUDE_DIRS = {
    "models",
    "__pycache__",
    ".pytest_cache",
    ".git",
    "demo/_tts_tmp",
    "demo/_tts_tmp_v2",
    "output/_video_tmp",
}
EXCLUDE_EXTS = {".pyc", ".tmp", ".bak", ".wav", ".mp3"}
EXCLUDE_FILES = {
    "pyrightconfig.json",
    "_zzz_c_site_packages.pth",
    "_zzz_d_site_packages.pth",
}

count = 0
seen = set()  # deduplicate by content hash

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT).replace(os.sep, "/")
        # Filter dirs in-place
        filtered = []
        for d in dirnames:
            full = f"{rel}/{d}" if rel != "." else d
            if full not in EXCLUDE_DIRS and d not in EXCLUDE_DIRS:
                filtered.append(d)
        dirnames[:] = filtered
        for fn in sorted(filenames):
            if any(fn.endswith(ext) for ext in EXCLUDE_EXTS):
                continue
            if fn in EXCLUDE_FILES:
                continue
            fp = os.path.join(dirpath, fn)
            arcname = os.path.relpath(fp, os.path.dirname(ROOT)).replace(os.sep, "/")
            # Deduplicate by file content hash
            with open(fp, "rb") as f:
                content_hash = hashlib.md5(f.read()).hexdigest()
            dedup_key = (arcname, content_hash)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            zf.write(fp, arcname)
            count += 1

size_mb = os.path.getsize(OUT) / (1024 * 1024)
print(f"Files: {count}")
print(f"ZIP: {OUT}")
print(f"Size: {size_mb:.1f} MB")
