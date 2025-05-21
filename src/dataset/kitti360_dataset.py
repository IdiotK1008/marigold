# Last modified: 2024-02-08
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
import os
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class Kitti360Dataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # Kitti360 data parameter
            min_depth=1e-5,
            max_depth=65.0,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        image_to_read = os.path.join(self.dataset_dir, rel_path)
        data = np.load(image_to_read)

        # 提取稀疏矩阵的组件
        indices = data['indices']
        indptr = data['indptr']
        shape = data['shape']
        format = data['format']
        data_values = data['data']

        # 从 NumPy 数组中提取字节字符串并解码为普通字符串
        if isinstance(format, np.ndarray):
            format = format.item()  # 提取数组中的值（字节字符串）
            if isinstance(format, bytes):
                format = format.decode('utf-8')  # 将字节字符串解码为普通字符串
                
        # 根据 format 键的值确定稀疏矩阵格式
        if format == 'csr':
            sparse_matrix = csr_matrix((data_values, indices, indptr), shape=shape)
        elif format == 'csc':
            sparse_matrix = csc_matrix((data_values, indices, indptr), shape=shape)
        else:
            raise ValueError(f"Unsupported sparse matrix format: {format}")

        # 将稀疏矩阵转换为密集矩阵（如果需要）
        depth_decoded = sparse_matrix.toarray()

        return depth_decoded
