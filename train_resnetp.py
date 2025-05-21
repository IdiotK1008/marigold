import argparse
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.util.dmvnp_resnet import ResnetDepthMeanVarianceNetPlus # resnet18
from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler

from torch.optim import Adam
from torch.nn import MSELoss

# Configure logging
t_start = datetime.now()
log_filename = f"train_resnet_{t_start.strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join("./logs", log_filename)

os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)

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

invalid_list = [
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000000_24800_346867.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000010_25287_381346.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000020_24739_320437.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000030_24322_312806.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000040_23443_304821.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000050_23147_299581.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000060_24074_297697.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000070_23029_306894.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000080_23229_302906.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000090_23305_292783.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000100_24614_305821.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000110_24676_301610.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000120_24902_300995.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000130_24017_298173.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000140_24231_297082.npy",
    "./my_data/train_fisheye/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000150_23957_283424.npy"
]

# Custom dataset class
class DepthLatentDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = self._get_file_list(data_dir)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # Load data
        latent_data = np.load(file_path)  # [C, H, W]
        latent_data = torch.tensor(latent_data).float()  # Convert to PyTorch tensor
        
        # Extract mean and variance from filename
        file_name = os.path.basename(file_path)
        parts = file_name.split("_")
        mean = float(parts[-2]) / 1000.0  # Second last part is mean
        variance = float(parts[-1].replace(".npy", "")) / 1000.0  # Last part is variance

        # Camera parameters
        if "image_02" in file_path:
            camera_params = kitti360_camera["image_02"]
        elif "image_03" in file_path:
            camera_params = kitti360_camera["image_03"]
        else:
            print("filename: ", file_path)
            camera_params = [0.0, 0.0, 0.0, 0.0]  # Default camera parameters if not found
        camera_params = torch.tensor(camera_params).float()  # Convert to PyTorch tensor
        
        return latent_data, torch.tensor([mean]), torch.tensor([variance]), camera_params
    
    def _get_file_list(self, data_dir):
        """
        Recursively get all .npy files in data_dir and its subfolders.
        """
        file_list = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".npy"):
                    file_list.append(os.path.join(root, file))
        print("len: ", len(file_list))
        valid_file_list = [f for f in file_list if f not in invalid_list]
        print("len: ", len(valid_file_list))
        return valid_file_list

# Training function
def train_dmvn(model, data_dir, output_dir, num_epochs=10, batch_size=32, lr=1e-4, max_iter=None, mean_weight=0.9, device=None):
    # Define loss function and optimizer
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Load datasets
    data_dir = "./my_data/train_fisheye"
    dataset = DepthLatentDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    iteration = 0
    min_loss = 1

    # Training loop
    # for epoch in range(num_epochs):
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        model.train()
        epoch_loss = 0.0
        
        for latent_data, mean_gt, variance_gt, camera in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data to device
            latent_data = latent_data.to(device)  # [B, C, H, W]
            mean_gt = mean_gt.to(device)  # [B, 1]
            variance_gt = variance_gt.to(device)  # [B, 1]
            camera = camera.to(device)  # [B, 4]

            # Forward pass
            pred_mean, pred_variance = model(latent_data, camera)

            # Calculate loss
            mean_loss = criterion(pred_mean, mean_gt)
            variance_loss = criterion(pred_variance, variance_gt)
            total_loss = mean_weight * mean_loss + (1 - mean_weight) * variance_loss

            # Backward pass and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            iteration += 1

            # Check if max iteration reached
            if max_iter is not None and iteration >= max_iter:
                logging.info(f"Reached max iteration {max_iter}. Stopping training.")
                break
        
        # Save model if loss is 90% of minimum loss
        '''if (epoch_loss/len(dataloader)) < 0.9 * min_loss:
            min_loss = epoch_loss
            model_save_path = os.path.join(output_dir, f"checkpoints/depth_mean_variance_net_{round(epoch_loss*10000)}.pth")
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model saved to {model_save_path}")'''
        
        # Save model every 50 epochs
        if (epoch+1) % 10 == 0:
            model_save_path = os.path.join(output_dir, f"checkpoints/depth_mean_variance_net_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model saved to {model_save_path}")

        # Check if max iteration reached
        if max_iter is not None and iteration >= max_iter:
            break
        
        # Log epoch loss
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "dmvnp_100.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Final model saved to {model_save_path}")

if "__main__" == __name__:
    t_start = datetime.now()
    logging.info(f"Training started at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--output", 
        type=str, 
        default="./base_ckpt_dir/dmvn_resnet",
        help="directory to save checkpoints"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="./my_data/train",
        help="directory of training data"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./base_ckpt_dir/dmvn_resnet/dmvnp_origin.pth",
        help="directory of pretrained pipeline checkpoint",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="maximum number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate for training"
    )
    parser.add_argument(
        "--max_iter", type=int, default=None, help="maximum number of iterations"
    )
    parser.add_argument(
        "--mean_weight", type=int, default=0.9, help="weight for mean loss"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    output_dir = args.output
    data_dir = args.data
    ckpt_path = args.checkpoint

    num_epochs = args.max_epochs
    batch_size = args.batch_size
    lr = args.lr
    max_iter = args.max_iter
    mean_weight = args.mean_weight
    seed = args.seed

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"Using device: {device}")

    # -------------------- Model --------------------
    model = ResnetDepthMeanVarianceNetPlus(input_channels=8)

    # Load pretrained model
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict, strict=False)

    # Freeze ViT parameters
    for name, param in model.named_parameters():
        if "feature_extractor" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    model.to(device)
    logging.info(f"Model loaded from {ckpt_path}")

    # -------------------- Train --------------------
    train_dmvn(model, data_dir, output_dir, num_epochs, batch_size, lr, max_iter, mean_weight, device)
    
    t_end = datetime.now()
    logging.info(f"Training completed at {t_end}")
    logging.info(f"Total training time: {t_end - t_start}")