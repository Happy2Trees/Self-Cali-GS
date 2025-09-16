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
from argparse import ArgumentParser
import shutil
from typing import Dict, List

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

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
