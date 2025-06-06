# Last modified: 2024-04-30
#
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
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import io
import os
import random
import tarfile
from enum import Enum
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize
import yaml

from src.util.depth_transform import DepthNormalizerBase


class DatasetMode(Enum):
    RGB_ONLY = "rgb_only"
    EVAL = "evaluate"
    TRAIN = "train"


class DepthFileNameMode(Enum):
    """Prediction file naming modes"""

    id = 1  # id.png
    rgb_id = 2  # rgb_id.png
    i_d_rgb = 3  # i_d_1_rgb.png
    rgb_i_d = 4


def read_image_from_tar(tar_obj, img_rel_path):
    image = tar_obj.extractfile("./" + img_rel_path)
    image = image.read()
    image = Image.open(io.BytesIO(image))


class BaseDepthDataset(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        dataset_dir: str,
        disp_name: str,
        min_depth: float,
        max_depth: float,
        has_filled_depth: bool,
        name_mode: DepthFileNameMode,
        depth_transform: Union[DepthNormalizerBase, None] = None,
        augmentation_args: dict = None,
        resize_to_hw=None,
        move_invalid_to_far_plane: bool = True,
        rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        # dataset info
        self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset does not exist at: {self.dataset_dir}"
        self.disp_name = disp_name
        self.has_filled_depth = has_filled_depth
        self.name_mode: DepthFileNameMode = name_mode
        self.min_depth = min_depth
        self.max_depth = max_depth

        # training arguments
        self.depth_transform: DepthNormalizerBase = depth_transform
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw
        self.rgb_transform = rgb_transform
        self.move_invalid_to_far_plane = move_invalid_to_far_plane

        # Load filenames
        with open(self.filename_ls_path, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]  # [['rgb.png', 'depth.tif'], [], ...]

        # Tar dataset
        self.tar_obj = None
        self.is_tar = (
            True
            if os.path.isfile(dataset_dir) and tarfile.is_tarfile(dataset_dir)
            else False
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
            # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
            )
            # rasters = {k: resize_transform(v) for k, v in rasters.items()}
            rasters = {k: self._resize_camera_data(v, rasters["rgb_int"]) if k == "camera" else resize_transform(v) for k, v in rasters.items()}
        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, filled_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=filled_rel_path
            )
            rasters.update(depth_data)
            # valid mask
            rasters["valid_mask_raw"] = self._get_valid_mask(
                rasters["depth_raw_linear"]
            ).clone()
            rasters["valid_mask_filled"] = self._get_valid_mask(
                rasters["depth_filled_linear"]
            ).clone()

        # Camera data
        rasters["camera"] = self._get_camera_data(path = rgb_rel_path)

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other
    
    def _get_camera_data(self, path):
        if self.disp_name.startswith("kitti360"):
            # Read camera data
            camera_path = self.dataset_dir.replace("data_2d_raw", "calibration")
            if "image_02" in path:
                camera_path = os.path.join(camera_path, "image_02.yaml")
            elif "image_03" in path:
                camera_path = os.path.join(camera_path, "image_03.yaml")
            else:
                print("path: ", path)
                raise NotImplementedError
            with open(camera_path, "r") as f:
                camera_data_all = yaml.safe_load(f)
            camera_data = [camera_data_all["projection_parameters"]["gamma1"], camera_data_all["projection_parameters"]["gamma2"]]
        elif self.disp_name.startswith("synwoodscape"):
            sws_camera = {
                "FV": [772.0, 771.8, 0.33, -3.55],
                "MVL": [772.0, 771.9, 1.185, -3.37],
                "MVR": [772.0, 771.6, 0.384, -0.461],
                "RV": [772.0, 771.7, 2.86472, -2.06627],
            }
            if "FV" in path:
                camera_data = [772.0, 771.8]
            elif "MVL" in path:
                camera_data = [772.0, 771.9]
            elif "MVR" in path:
                camera_data = [772.0, 771.6]
            elif "RV" in path:
                camera_data = [772.0, 771.7]
            else:
                print("path: ", path)
                raise NotImplementedError
        else:
            camera_data = [0.0, 0.0]
        camera_data = torch.tensor(camera_data).float() # [2]
        return camera_data
    
    def _resize_camera_data(self, camera_data, rgb_int):
        if camera_data is not None:
            new_height = self.resize_to_hw[0]
            new_width = self.resize_to_hw[1]
            old_height = rgb_int.shape[1]
            old_width = rgb_int.shape[2]
            if old_height != new_height or old_width != new_width:
                # Resize camera data
                camera_data[0] = camera_data[0] * new_width / old_width
                camera_data[1] = camera_data[1] * new_height / old_height
        return camera_data

    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        outputs = {
            "rgb_int": torch.from_numpy(rgb).int(),
            "rgb_norm": torch.from_numpy(rgb_norm).float(),
        }
        return outputs

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        # Read depth data
        outputs = {}
        depth_raw = self._read_depth_file(depth_rel_path).squeeze()
        depth_raw_linear = torch.from_numpy(depth_raw).float().unsqueeze(0)  # [1, H, W]
        outputs["depth_raw_linear"] = depth_raw_linear.clone()

        if self.has_filled_depth:
            depth_filled = self._read_depth_file(filled_rel_path).squeeze()
            depth_filled_linear = torch.from_numpy(depth_filled).float().unsqueeze(0)
            outputs["depth_filled_linear"] = depth_filled_linear
        else:
            outputs["depth_filled_linear"] = depth_raw_linear.clone()

        return outputs

    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        rgb_rel_path = filename_line[0]

        depth_rel_path, filled_rel_path = None, None
        if DatasetMode.RGB_ONLY != self.mode:
            depth_rel_path = filename_line[1]
            if self.has_filled_depth:
                filled_rel_path = filename_line[2]
        return rgb_rel_path, depth_rel_path, filled_rel_path

    def _read_image(self, img_rel_path) -> np.ndarray:
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            image_to_read = self.tar_obj.extractfile("./" + img_rel_path)
            image_to_read = image_to_read.read()
            image_to_read = io.BytesIO(image_to_read)
        else:
            image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        image = Image.open(image_to_read)  # [H, W, rgb]
        image = np.asarray(image)
        return image

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        rgb = self._read_image(rel_path)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
        return rgb

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        #  Replace code below to decode depth according to dataset definition
        depth_decoded = depth_in

        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and(
            (depth > self.min_depth), (depth < self.max_depth)
        ).bool()
        return valid_mask

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        # log depth: log(d_r/d_min)/log(d_max/d_min)
        log_max = 80.0
        log_min = 0.5
        rasters["depth_log_raw"] = rasters["depth_filled_linear"].clone()
        rasters["depth_log_raw"] = torch.clamp(
            rasters["depth_log_raw"], min=log_min, max=log_max
        )
        rasters["depth_log_raw"] = torch.log(rasters["depth_log_raw"] / log_min) / torch.log(torch.tensor(log_max / log_min))

        # Normalization
        rasters["depth_raw_norm"] = self.depth_transform(
            rasters["depth_raw_linear"], rasters["valid_mask_raw"]
        ).clone()
        rasters["depth_filled_norm"] = self.depth_transform(
            rasters["depth_filled_linear"], rasters["valid_mask_filled"]
        ).clone()
        rasters["depth_log_norm"] = self.depth_transform(
            rasters["depth_log_raw"], rasters["valid_mask_filled"]
        ).clone()

        # one over minmax mode
        rasters["valid_mask_one_over"] = torch.logical_and(
            rasters["depth_raw_linear"] > 0.5, rasters["depth_raw_linear"] < 80.0
        ).bool()
        rasters["depth_one_over"] = 1.0 / torch.clamp(
            rasters["depth_raw_linear"], 0.5, 80.0
        )
        '''rasters["depth_one_over_norm"] = self.depth_transform(
            rasters["depth_one_over"], rasters["valid_mask_one_over"]
        ).clone()'''
        # fixer normalize
        rasters["depth_one_over_norm"] = rasters["depth_one_over"].clone()
        fixed_min = 1.0 / 80.0
        fixed_max = 1.0 / 0.5
        rasters["depth_one_over_norm"] = (rasters["depth_one_over_norm"] - fixed_min) / (fixed_max - fixed_min)
        rasters["depth_one_over_norm"] = rasters["depth_one_over_norm"] * 2.0 - 1.0 # [-1, 1]

        # Set invalid pixel to far plane
        if self.move_invalid_to_far_plane:
            if self.depth_transform.far_plane_at_max:
                rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_max
                )
                rasters["depth_one_over_norm"][~rasters["valid_mask_one_over"]] = 1.0
            else:
                rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_min
                )
                rasters["depth_one_over_norm"][~rasters["valid_mask_one_over"]] = -1.0

        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
            )
            # rasters = {k: resize_transform(v) for k, v in rasters.items()}
            rasters = {k: self._resize_camera_data(v, rasters["rgb_int"]) if k == "camera" else resize_transform(v) for k, v in rasters.items()}

        return rasters

    def _augment_data(self, rasters_dict):
        # lr flipping
        lr_flip_p = self.augm_args.lr_flip_p
        if random.random() < lr_flip_p:
            # rasters_dict = {k: v.flip(-1) for k, v in rasters_dict.items()}
            rasters_dict = {k: v.flip(-1) if k != "camera" else v for k, v in rasters_dict.items()}

        return rasters_dict

    def __del__(self):
        if hasattr(self, "tar_obj") and self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None


def get_pred_name(rgb_basename, name_mode, suffix=".png"):
    if DepthFileNameMode.rgb_id == name_mode:
        pred_basename = "pred_" + rgb_basename.split("_")[1]
    elif DepthFileNameMode.i_d_rgb == name_mode:
        pred_basename = rgb_basename.replace("_rgb.", "_pred.")
    elif DepthFileNameMode.id == name_mode:
        pred_basename = "pred_" + rgb_basename
    elif DepthFileNameMode.rgb_i_d == name_mode:
        pred_basename = "pred_" + "_".join(rgb_basename.split("_")[1:])
    else:
        raise NotImplementedError
    # change suffix
    pred_basename = os.path.splitext(pred_basename)[0] + suffix

    return pred_basename
