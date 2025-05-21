import os
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class SynWoodScapeDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # SynWoodScape data parameter
            min_depth=1e-5,
            max_depth=40.0,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        image_to_read = os.path.join(self.dataset_dir, rel_path)
        data = np.load(image_to_read)

        return data