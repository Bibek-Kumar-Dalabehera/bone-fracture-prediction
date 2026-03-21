"""
prepare_data.py
───────────────
Stage 1 — Mendeley only (Normal vs Fracture)
  • Split by original image ID so all 7 versions (1 original + 6 augmentations)
    stay in the same split → prevents data leakage.
  • Fracture = simple_fracture + major_fracture merged.

Stage 2 — pkdarabi only (hairline vs major_crack)
  • hairline  → pkdarabi "Hairline Fracture"
  • major_crack → pkdarabi "Comminuted fracture" + "Oblique fracture"
                          + "Spiral Fracture" + "Transverse fracture" (if exists)
  • Random 70 / 15 / 15 split (no patient IDs available).
"""

import os
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict
# tqdm is optional — works without it
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

random.seed(42)

# ─── paths ────────────────────────────────────────────────────────────────────
BASE     = Path("/home/user/Desktop/bone_fracture_project")
RAW      = BASE / "data" / "raw"

MEND_ORIG = RAW / "Mendeley" / "Bone_Fracture_Dataset" / "Final Dataset" / "original"
MEND_AUG  = RAW / "Mendeley" / "Bone_Fracture_Dataset" / "Final Dataset" / "augmented"

PKDARABI  = (RAW / "pkdarabi" / "Bone Break Classification"
                 / "Bone Break Classification" / "Bone Break Classification")

STAGE1    = BASE / "data" / "stage1"
STAGE2    = BASE / "data" / "stage2"


# ─── helpers ──────────────────────────────────────────────────────────────────
def split_list(items, train=0.70, val=0.15):
    """Split a list into train / val / test."""
    random.shuffle(items)
    n = len(items)
    t = int(n * train)
    v = int(n * (train + val))
    return items[:t], items[t:v], items[v:]


def copy_files(file_list, dst_dir):
    """Copy a list of Path objects into dst_dir (flat)."""
    os.makedirs(dst_dir, exist_ok=True)
    for f in file_list:
        shutil.copy2(str(f), str(dst_dir / f.name))


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1  —  Mendeley: Normal vs Fracture
# ═══════════════════════════════════════════════════════════════════════════════
def extract_image_id(filename):
    """
    Mendeley filenames look like:
      Original : 42_hand_normal_1.png
      Augmented: 42_hand_normal_1_HFlip.png
    The "image ID" is the leading integer (e.g. 42).
    All 7 variants of image 42 share this ID.
    """
    m = re.match(r"^(\d+)_", filename)
    return m.group(1) if m else filename


def gather_mendeley_by_id(orig_dir, aug_dir):
    """
    Returns {image_id: [Path, Path, …]} grouping original + augmented.
    """
    groups = defaultdict(list)
    for d in [orig_dir, aug_dir]:
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                gid = extract_image_id(f.name)
                groups[gid].append(f)
    return groups


def prepare_stage1():
    print("\n══════ STAGE 1 : Normal vs Fracture (Mendeley) ══════")

    # ── normal ────────────────────────────────────────────────────────────────
    normal_groups = gather_mendeley_by_id(
        MEND_ORIG / "normal",
        MEND_AUG  / "normal",
    )
    normal_ids = sorted(normal_groups.keys())
    print(f"Normal unique IDs : {len(normal_ids)}")

    # ── fracture  (simple + major merged) ─────────────────────────────────────
    frac_groups = defaultdict(list)
    for subtype in ("simple_fracture", "major_fracture"):
        sub = gather_mendeley_by_id(
            MEND_ORIG / "fracture" / subtype,
            MEND_AUG  / "fracture" / subtype,
        )
        for gid, files in sub.items():
            frac_groups[f"{subtype}_{gid}"].extend(files)
    frac_ids = sorted(frac_groups.keys())
    print(f"Fracture unique IDs: {len(frac_ids)}")

    # ── split by image‑group ID ───────────────────────────────────────────────
    for label, id_list, groups in [
        ("normal",   normal_ids, normal_groups),
        ("fracture", frac_ids,   frac_groups),
    ]:
        train_ids, val_ids, test_ids = split_list(id_list)
        for split_name, ids in [("train", train_ids),
                                ("val",   val_ids),
                                ("test",  test_ids)]:
            dst = STAGE1 / split_name / label
            files = [f for gid in ids for f in groups[gid]]
            copy_files(files, dst)
            print(f"  {split_name}/{label}: {len(files)} images")


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2  —  pkdarabi: hairline vs major_crack (2 classes)
# ═══════════════════════════════════════════════════════════════════════════════
# Class mapping
S2_MAP = {
    "Hairline Fracture":    "hairline",
    "Comminuted fracture":  "major_crack",
    "Oblique fracture":     "major_crack",
    "Spiral Fracture":      "major_crack",
    "Transverse fracture":  "major_crack",   # may or may not exist in this dataset
}


def prepare_stage2():
    print("\n══════ STAGE 2 : hairline vs major_crack (pkdarabi) ══════")

    class_files = defaultdict(list)
    for folder_name, target_class in S2_MAP.items():
        src = PKDARABI / folder_name
        if not src.exists():
            print(f"  [skip] {folder_name} — folder not found")
            continue
        # pkdarabi has Train/ and Test/ subdirs inside each class folder
        # so we need to walk recursively
        imgs = [f for f in src.rglob("*")
                if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        class_files[target_class].extend(imgs)
        print(f"  {folder_name} → {target_class}: {len(imgs)} images")

    for cls, files in class_files.items():
        train_f, val_f, test_f = split_list(files)
        for split_name, flist in [("train", train_f),
                                  ("val",   val_f),
                                  ("test",  test_f)]:
            dst = STAGE2 / split_name / cls
            copy_files(flist, dst)
            print(f"  {split_name}/{cls}: {len(flist)} images")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    prepare_stage1()
    prepare_stage2()

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n══════ SUMMARY ══════")
    for stage_dir in (STAGE1, STAGE2):
        print(f"\n{stage_dir.name}/")
        for split in ("train", "val", "test"):
            sd = stage_dir / split
            if not sd.exists():
                continue
            for cls in sorted(os.listdir(sd)):
                cd = sd / cls
                n = len([f for f in cd.iterdir() if f.is_file()])
                print(f"  {split}/{cls}: {n}")
