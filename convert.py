#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
import sqlite3
from argparse import ArgumentParser
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import imageio.v2 as imageio

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--mask_radius", default=None, type=float,
                    help="Optional circular mask radius in pixels applied around the image center.")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root_dir: str) -> List[Tuple[str, str]]:
    """Return (absolute_path, relative_path) pairs for supported images under root_dir."""
    image_entries: List[Tuple[str, str]] = []
    for current_root, _, files in os.walk(root_dir):
        rel_root = os.path.relpath(current_root, root_dir)
        for filename in files:
            if os.path.splitext(filename)[1].lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            absolute_path = os.path.join(current_root, filename)
            relative_path = filename if rel_root == os.curdir else os.path.join(rel_root, filename)
            image_entries.append((absolute_path, relative_path))
    return sorted(image_entries, key=lambda entry: entry[1])


def create_circular_mask(height: int, width: int, radius_px: float) -> np.ndarray:
    """Create a binary circular mask centered in the image with radius_px."""
    yy, xx = np.ogrid[:height, :width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    squared_distance = (yy - center_y) ** 2 + (xx - center_x) ** 2
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[squared_distance <= radius_px ** 2] = 255
    return mask


def generate_masks(image_root: str, output_root: str, radius_px: float) -> None:
    """Generate circular masks for all supported images under image_root."""
    images = list_images(image_root)
    if not images:
        logging.warning("No images found under '%s'; skipping mask generation.", image_root)
        return

    os.makedirs(output_root, exist_ok=True)

    generated_count = 0
    skipped_count = 0

    for abs_path, rel_path in images:
        try:
            image = imageio.imread(abs_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning("Failed to read '%s' for mask generation: %s", abs_path, exc)
            skipped_count += 1
            continue

        if image.ndim < 2:
            logging.warning("Unsupported image dimensions for '%s'; skipping mask generation.", abs_path)
            skipped_count += 1
            continue

        height, width = image.shape[:2]
        effective_radius = min(radius_px, min(height, width) / 2.0)

        if effective_radius <= 0:
            logging.warning("Non-positive effective radius for '%s'; skipping mask generation.", abs_path)
            skipped_count += 1
            continue

        mask = create_circular_mask(height, width, effective_radius)

        rel_directory = os.path.dirname(rel_path)
        output_directory = output_root if not rel_directory else os.path.join(output_root, rel_directory)
        os.makedirs(output_directory, exist_ok=True)

        mask_filename = os.path.basename(rel_path)
        output_path = os.path.join(output_directory, mask_filename)

        try:
            imageio.imwrite(output_path, mask)
            generated_count += 1
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning("Failed to write mask '%s': %s", output_path, exc)
            skipped_count += 1

    logging.info("Generated %d masks under '%s' (skipped %d).", generated_count, output_root, skipped_count)


def resolve_mask_path(mask_root: str, image_name: str) -> Optional[str]:
    """Return the filesystem path to a mask image for the given COLMAP image name."""
    candidate_png = os.path.join(mask_root, image_name + ".png")
    if os.path.isfile(candidate_png):
        return candidate_png

    candidate_same = os.path.join(mask_root, image_name)
    if os.path.isfile(candidate_same):
        return candidate_same

    return None


def apply_mask_to_database(database_path: str, mask_root: str) -> None:
    """Filter COLMAP keypoints/descriptors according to binary masks."""
    if not os.path.isfile(database_path):
        logging.error("COLMAP database not found at '%s'; cannot apply masks.", database_path)
        return

    conn = sqlite3.connect(database_path)
    cur = conn.cursor()

    try:
        cur.execute("SELECT image_id, name FROM images")
        images = cur.fetchall()
    except sqlite3.Error as exc:
        conn.close()
        logging.error("Failed to enumerate images in database '%s': %s", database_path, exc)
        return

    total_removed = 0
    processed_images = 0

    for image_id, image_name in images:
        mask_path = resolve_mask_path(mask_root, image_name)
        if mask_path is None:
            logging.warning("Mask not found for '%s'; skipping database filtering for this image.", image_name)
            continue

        try:
            mask = imageio.imread(mask_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning("Failed to load mask '%s': %s", mask_path, exc)
            continue

        if mask.ndim == 3:
            mask = mask[..., 0]

        try:
            cur.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,))
            kp_row = cur.fetchone()
        except sqlite3.Error as exc:
            logging.warning("Failed to read keypoints for '%s': %s", image_name, exc)
            continue

        if not kp_row:
            continue

        kp_rows, kp_cols, kp_blob = kp_row
        if kp_rows == 0 or not kp_blob:
            continue

        keypoints = np.frombuffer(kp_blob, dtype=np.float32).reshape(kp_rows, kp_cols)

        try:
            cur.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (image_id,))
            desc_row = cur.fetchone()
        except sqlite3.Error as exc:
            logging.warning("Failed to read descriptors for '%s': %s", image_name, exc)
            continue

        if not desc_row:
            continue

        desc_rows, desc_cols, desc_blob = desc_row
        if desc_rows != kp_rows or desc_cols <= 0:
            logging.warning("Descriptor mismatch for '%s'; skipping mask filtering.", image_name)
            continue

        descriptors = np.frombuffer(desc_blob, dtype=np.uint8).reshape(desc_rows, desc_cols)

        max_x = mask.shape[1] - 1
        max_y = mask.shape[0] - 1
        xs = np.clip(np.rint(keypoints[:, 0]).astype(int), 0, max_x)
        ys = np.clip(np.rint(keypoints[:, 1]).astype(int), 0, max_y)
        valid = mask[ys, xs] > 0

        kept = int(np.count_nonzero(valid))
        if kept == kp_rows:
            processed_images += 1
            continue

        filtered_keypoints = keypoints[valid]
        filtered_descriptors = descriptors[valid]

        cur.execute(
            "UPDATE keypoints SET rows=?, data=? WHERE image_id=?",
            (kept, filtered_keypoints.astype(np.float32).tobytes(), image_id),
        )
        cur.execute(
            "UPDATE descriptors SET rows=?, data=? WHERE image_id=?",
            (kept, filtered_descriptors.astype(np.uint8).tobytes(), image_id),
        )

        total_removed += kp_rows - kept
        processed_images += 1

    conn.commit()
    conn.close()

    logging.info(
        "Applied masks to database '%s': filtered %d images, removed %d keypoints.",
        database_path,
        processed_images,
        total_removed,
    )


mask_directory: Optional[str] = None
if args.mask_radius is not None:
    if args.mask_radius <= 0:
        logging.error("Mask radius must be positive when provided.")
        exit(1)

    mask_directory = os.path.join(args.source_path, "masks")
    logging.info("Creating circular masks (radius %.2f px) in '%s'.", args.mask_radius, mask_directory)
    generate_masks(os.path.join(args.source_path, "input"), mask_directory, args.mask_radius)

mask_argument = ""
if mask_directory is not None:
    mask_argument = (
        " \\\n        --ImageReader.mask_path " + mask_directory
    )

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + mask_argument + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    if mask_directory is not None:
        apply_mask_to_database(os.path.join(args.source_path, "distorted", "database.db"), mask_directory)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)


def list_numeric_directories(base_path: str) -> List[str]:
    """Return all numeric subdirectory names inside base_path sorted by index."""
    if not os.path.isdir(base_path):
        return []

    numeric_dirs: List[str] = []
    for entry in os.listdir(base_path):
        entry_path = os.path.join(base_path, entry)
        if os.path.isdir(entry_path) and entry.isdigit():
            numeric_dirs.append(entry)

    numeric_dirs.sort(key=int)
    return numeric_dirs


distorted_sparse_root = os.path.join(args.source_path, "distorted", "sparse")
reconstruction_paths: Dict[str, str] = {}
numeric_recon_dirs = list_numeric_directories(distorted_sparse_root)

if numeric_recon_dirs:
    for recon_dir in numeric_recon_dirs:
        reconstruction_paths[recon_dir] = os.path.join(distorted_sparse_root, recon_dir)
    latest_recon_id = numeric_recon_dirs[-1]
    undistort_input_path = reconstruction_paths[latest_recon_id]
else:
    # No explicit reconstruction folders were created; treat the root as reconstruction "0".
    latest_recon_id = "0"
    reconstruction_paths[latest_recon_id] = distorted_sparse_root
    undistort_input_path = distorted_sparse_root

if not os.path.isdir(undistort_input_path):
    logging.error(f"No reconstruction found at '{undistort_input_path}'. Exiting.")
    exit(1)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + undistort_input_path + " \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

sparse_root = os.path.join(args.source_path, "sparse")
os.makedirs(sparse_root, exist_ok=True)

existing_numeric_dirs = list_numeric_directories(sparse_root)
latest_sparse_dir = os.path.join(sparse_root, latest_recon_id)

# Ensure the undistorted reconstruction is stored under the latest reconstruction id.
if latest_recon_id in existing_numeric_dirs:
    # Already structured correctly, nothing to move for the latest reconstruction.
    existing_numeric_dirs.remove(latest_recon_id)
elif existing_numeric_dirs:
    # Move the numerically-last existing directory to the latest reconstruction id.
    source_dir_name = existing_numeric_dirs[-1]
    source_dir_path = os.path.join(sparse_root, source_dir_name)
    if os.path.exists(latest_sparse_dir):
        shutil.rmtree(latest_sparse_dir)
    shutil.move(source_dir_path, latest_sparse_dir)
else:
    # Move any loose files in sparse_root into the latest reconstruction directory.
    if os.path.exists(latest_sparse_dir):
        shutil.rmtree(latest_sparse_dir)
    os.makedirs(latest_sparse_dir, exist_ok=True)
    for entry in os.listdir(sparse_root):
        entry_path = os.path.join(sparse_root, entry)
        if entry == latest_recon_id:
            continue
        shutil.move(entry_path, os.path.join(latest_sparse_dir, entry))

# Copy remaining reconstructions (if any) from the distorted folder.
for recon_id, source_path in reconstruction_paths.items():
    if recon_id == latest_recon_id:
        continue

    destination_path = os.path.join(sparse_root, recon_id)
    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)
    shutil.copytree(source_path, destination_path)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
