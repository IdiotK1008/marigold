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

from marigold.marigold_pipeline import MarigoldPipeline
from src.util.dmvn_vit import VitDepthMeanVarianceNet # vit
from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler

from torch.optim import Adam
from torch.nn import MSELoss

# Configure logging
t_start = datetime.now()
log_filename = f"train_vit_{t_start.strftime('%Y%m%d_%H%M%S')}.log"
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

# Custom dataset class
class DepthLatentDataset(Dataset):
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.file_list = self._get_file_list(config_dir)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # Load data
        latent_data = np.load(file_path)  # [C, H, W]
        latent_data = torch.tensor(latent_data).float()  # Convert to PyTorch tensor
        rgb_file_path = "_".join(file_path.replace("train", "train_rgb").split("_")[:-2]) + ".npy"
        rgb_data = np.load(rgb_file_path)  # [1, C, H, W]
        rgb_data = rgb_data.squeeze(0)  # Remove the first dimension
        rgb_data = torch.tensor(rgb_data).float()  # Convert to PyTorch tensor
        
        # Concatenate latent and RGB data
        data = torch.cat((latent_data, rgb_data), dim=0)  # [C, H, W]
        
        '''# Adjust input size to be divisible by 16 (for ViT)
        _, H, W = data.shape
        new_H = ((H + 15) // 16) * 16  # Round up to nearest multiple of 16
        new_W = ((W + 15) // 16) * 16  # Round up to nearest multiple of 16
        
        if H != new_H or W != new_W:
            # Resize data using bilinear interpolation
            data = F.interpolate(data.unsqueeze(0), size=(new_H, new_W), mode="bilinear", align_corners=False)
            data = data.squeeze(0)'''
        
        # Resize data to [8, 224, 224] for ViT
        data = F.interpolate(data.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        
        # Extract mean and variance from filename
        file_name = os.path.basename(file_path)
        parts = file_name.split("_")
        mean = float(parts[-2]) / 1000.0  # Second last part is mean
        variance = float(parts[-1].replace(".npy", "")) / 1000.0  # Last part is variance
        
        return data, torch.tensor([mean]), torch.tensor([variance])
    
    def _get_file_list(self, config_dir):
        """
        Read file paths from a text file and return a list of file paths.
        """
        file_list = []
        with open(config_dir, 'r') as f:
            for line in f:
                file_path = line.strip()
                if os.path.exists(file_path):
                    file_list.append(file_path)
                else:
                    logging.warning(f"File not found: {file_path}")
        return file_list

# Training function
def train_dmvn(model, data_dir, output_dir, num_epochs=10, batch_size=32, lr=1e-4, max_iter=None, mean_weight=0.9, device="cuda"):
    # Define loss function and optimizer
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Load datasets
    data_dir_1 = "./data_split/dmvn/hypersim_0_10_0_50.txt"
    data_dir_2 = "./data_split/dmvn/vkitti_0_25_0_320.txt"
    dataset_1 = DepthLatentDataset(data_dir_1)
    dataset_2 = DepthLatentDataset(data_dir_2)
    dataloader_1 = DataLoader(dataset_1, batch_size=batch_size, shuffle=True)
    dataloader_2 = DataLoader(dataset_2, batch_size=batch_size, shuffle=True)

    iteration = 0
    min_loss = 1

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        model.train()
        epoch_loss = 0.0

        if epoch % 2 == 0:
            dataloader = dataloader_1
        else:
            dataloader = dataloader_2
        
        for latent_data, mean_gt, variance_gt in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data to device
            latent_data = latent_data.to(device)  # [B, C, H, W]
            mean_gt = mean_gt.to(device)  # [B, 1]
            variance_gt = variance_gt.to(device)  # [B, 1]

            # Forward pass
            pred_mean, pred_variance = model(latent_data)

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
        if (epoch+1) % 5 == 0:
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
    model_save_path = os.path.join(output_dir, "dmvn_40.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Final model saved to {model_save_path}")

def calculate_mean_variance_stats(data_dir):
    """
    计算两个数据集上所有样本的均值和方差的均值、最值。

    Args:
        data_dir_1 (str): 第一个数据集的路径（例如 hypersim）。
        data_dir_2 (str): 第二个数据集的路径（例如 vkitti）。

    Returns:
        dict: 包含均值和方差的统计信息的字典。
    """
    # 加载数据集
    dataset = DepthLatentDataset(data_dir)

    # 初始化列表存储均值和方差
    means = []
    variances = []

    # 遍历数据集
    for _, mean, variance in dataset:
        means.append(mean.item())
        variances.append(variance.item())

    # 计算统计信息
    mean_mean = np.mean(means)
    mean_variance = np.mean(variances)
    min_mean = np.min(means)
    max_mean = np.max(means)
    min_variance = np.min(variances)
    max_variance = np.max(variances)

    # 返回结果
    print(f"Mean Mean: {mean_mean}, max Mean: {max_mean}, min Mean: {min_mean}")
    print(f"Mean Variance: {mean_variance}, max Variance: {max_variance}, min Variance: {min_variance}")

if "__main__" == __name__:
    t_start = datetime.now()
    logging.info(f"Training started at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--output", 
        type=str, 
        default="./base_ckpt_dir/dmvn_vit",
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
        default="./base_ckpt_dir/dmvn_vit/dmvn_20.pth",
        help="directory of pretrained pipeline checkpoint",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=20, help="maximum number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="learning rate for training"
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
    model = VitDepthMeanVarianceNet(input_channels=8)

    # Load pretrained model
    filtered_state_dict = torch.load(ckpt_path)
    model_state_dict = model.state_dict()
    for k, v in filtered_state_dict.items():
        if k in model_state_dict:
            model_state_dict[k] = v
    model.load_state_dict(model_state_dict, strict=False)  # strict=False to ignore missing keys

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