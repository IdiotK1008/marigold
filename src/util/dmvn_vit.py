import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from torch.nn import init

class VitDepthMeanVarianceNet(nn.Module):
    def __init__(self, input_channels, hidden_dim=256):
        super(VitDepthMeanVarianceNet, self).__init__()
        
        # 使用 ViT 作为特征提取器
        self.feature_extractor = ViTModel.from_pretrained("./base_ckpt_dir/dmvn_vit/origin")
        
        '''# 替换第一层卷积，适配输入通道数
        self.feature_extractor.embeddings.patch_embeddings.projection = nn.Conv2d(
            input_channels, 768, kernel_size=16, stride=16, padding=0
        )'''

        self.additional_conv = nn.Conv2d(input_channels, 3, kernel_size=1, stride=1, padding=0)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc_mean = nn.Linear(768, 1)  # ViT 输出特征维度为 768
        self.fc_variance = nn.Linear(768, 1)
    
    def initialize_weights(self):
        # 使用 Kaiming 初始化卷积层的权重
        init.kaiming_normal_(self.additional_conv.weight, mode='fan_in', nonlinearity='relu')
        init.constant_(self.additional_conv.bias, 0)
        # 使用 Kaiming 初始化全连接层的权重
        init.kaiming_normal_(self.fc_mean.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc_variance.weight, mode='fan_in', nonlinearity='relu')
        # 初始化偏置为 0
        init.constant_(self.fc_mean.bias, 0)
        init.constant_(self.fc_variance.bias, 0)
    
    def forward(self, x):
        # 额外的卷积层，适配输入通道数
        x = self.additional_conv(x)  # (batch_size, 3, H, W)
        # 特征提取
        outputs = self.feature_extractor(x)
        x = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        # 全局平均池化
        x = self.global_pool(x.transpose(1, 2))  # (batch_size, 768, 1)
        x = x.view(x.size(0), -1)
        # 输出均值和方差
        mean = self.fc_mean(x)
        variance = F.softplus(self.fc_variance(x))  # 使用 Softplus 确保方差非负
        return mean, variance

if __name__ == "__main__":
    # 初始化模型
    input_channels = 8
    model = VitDepthMeanVarianceNet(input_channels)
    
    '''# 加载本地 ViT 权重文件
    weight_path = "/base_ckpt_dir/vit/vit-base-patch16-224.bin"
    model.feature_extractor = ViTModel.from_pretrained(weight_path)'''
    
    '''# 冻结 ViT 的所有参数
    for param in model.feature_extractor.parameters():
        param.requires_grad = False'''
    
    '''# 冻结 ViT 的所有参数，除了 patch_embeddings.projection
    for name, param in model.feature_extractor.named_parameters():
        if "embeddings.patch_embeddings.projection" not in name:  # 保留 projection 可训练
            param.requires_grad = False'''
    
    # 初始化全连接层权重
    model.initialize_weights()
    
    # 仅保存 ViT 模型以外的权重
    state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if "feature_extractor" not in k}
    save_path = "./base_ckpt_dir/dmvn_vit/dmvn_origin.pth"
    torch.save(filtered_state_dict, save_path)
    print(f"Model weights saved to {save_path}")