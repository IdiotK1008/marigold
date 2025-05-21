# Last modified: 2024-05-24
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import argparse
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from marigold.vit_marigold_pipeline import VitMarigoldPipeline
from marigold.resnet_marigold_pipeline import ResnetMarigoldPipeline
from src.util.seeding import seed_all
from src.dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)
from src.util.dmvn_vit import VitDepthMeanVarianceNet
from src.util.dmvn_resnet import ResnetDepthMeanVarianceNet
from src.util.dmvnp_vit import VitDepthMeanVarianceNetPlus
from src.util.dmvnp_resnet import ResnetDepthMeanVarianceNetPlus

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint/marigold-v1-0",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--dmvn_model",
        type=str,
        choices=["resnet18", "vit"],
        default="vit",
        help="Model type for DMVN.",
    )
    parser.add_argument(
        "--dmvn_plus",
        type=bool,
        default=True,
        help="Whether use camera data.",
    )
    parser.add_argument(
        "--dmvn_checkpoint",
        type=str,
        default="./base_ckpt_dir/dmvn_vit/checkpoints/depth_mean_variance_net_epoch_5.pth",
        help="Checkpoint path or hub name.",
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=0,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        type=str,
        default="bilinear",
        help="Resampling method used to resize images. This can be one of 'bilinear' or 'nearest'.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    dmvn_model = args.dmvn_model
    dmvn_plus = args.dmvn_plus
    dmvn_path = args.dmvn_checkpoint
    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    seed = args.seed

    print(f"arguments: {args}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps}, ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res}, seed = {seed}; "
        f"dataset config = `{dataset_config}`."
    )

    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    def check_directory(directory):
        if os.path.exists(directory):
            response = (
                input(
                    f"The directory '{directory}' already exists. Are you sure to continue? (y/n): "
                )
                .strip()
                .lower()
            )
            if "y" == response:
                pass
            elif "n" == response:
                print("Exiting...")
                exit()
            else:
                print("Invalid input. Please enter 'y' (for Yes) or 'n' (for No).")
                check_directory(directory)  # Recursive call to ask again

    check_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.RGB_ONLY
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    if dmvn_model == "resnet18":
        MarigoldPipeline = ResnetMarigoldPipeline
        if dmvn_plus:
            DepthMeanVarianceNet = ResnetDepthMeanVarianceNetPlus
        else:
            DepthMeanVarianceNet = ResnetDepthMeanVarianceNet
    elif dmvn_model == "vit":
        MarigoldPipeline = VitMarigoldPipeline
        if dmvn_plus:
            DepthMeanVarianceNet = VitDepthMeanVarianceNetPlus
        else:
            DepthMeanVarianceNet = VitDepthMeanVarianceNet
    pipe = MarigoldPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )
    
    if dmvn_path is not None:
        dmvn = DepthMeanVarianceNet(input_channels=8)
        dmvn.load_state_dict(torch.load(dmvn_path))
        pipe.load_dmvnet(dmvn)
    else:
        pipe.load_dmvnet(None)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    pipe = pipe.to(device)
    logging.info(
        f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
    )

    # -------------------- Inference and saving --------------------
    sws_camera = {
        "FV": [772.0, 771.8, 0.33, -3.55],
        "MVL": [772.0, 771.9, 1.185, -3.37],
        "MVR": [772.0, 771.6, 0.384, -0.461],
        "RV": [772.0, 771.7, 2.86472, -2.06627],
    }
    kitti360_camera = {
        "image_02": [1336.3, 1335.8, 16.9, 05.8],
        "image_03": [1485.4, 1484.9, -6.0, -6.0],
    }
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Inferencing on {dataset.disp_name}", leave=True
        ):
            # Read input image
            rgb_int = batch["rgb_int"].squeeze().numpy().astype(np.uint8)  # [3, H, W]
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            input_image = Image.fromarray(rgb_int)
            file_name = batch["rgb_relative_path"]
            if "iamge_02" in file_name:
                camera_params = kitti360_camera["image_02"]
            elif "image_03" in file_name:
                camera_params = kitti360_camera["image_03"]
            else:
                camera_params = [0.0, 0.0, 0.0, 0.0]
            camera_params = torch.tensor(camera_params).float().unsqueeze(0)

            # Predict depth
            pipe_out = pipe(
                input_image,
                #dmvn_plus=dmvn_plus,
                camera_data=camera_params,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=0,
                color_map=None,
                show_progress_bar=False,
                resample_method=resample_method,
            )

            # Save predictions
            # depth_pred: np.ndarray = pipe_out.depth_np
            rgb_filename = batch["rgb_relative_path"][0]
            rgb_basename = os.path.basename(rgb_filename)
            scene_dir = os.path.join(output_dir, os.path.dirname(rgb_filename))
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            '''pred_basename = get_pred_name(
                rgb_basename, dataset.name_mode, suffix=".npy"
            )
            save_to = os.path.join(scene_dir, pred_basename)
            if os.path.exists(save_to):
                logging.warning(f"Existing file: '{save_to}' will be overwritten")

            np.save(save_to, depth_pred)'''

            # save abs depth
            abs_depth_pred: np.ndarray = pipe_out.abs_depth_np
            abs_pred_basename = get_pred_name(
                rgb_basename, dataset.name_mode, suffix="_abs.npy"
            )
            save_to = os.path.join(scene_dir, abs_pred_basename)
            if os.path.exists(save_to):
                logging.warning(f"Existing file: '{save_to}' will be overwritten")

            np.save(save_to, abs_depth_pred)
