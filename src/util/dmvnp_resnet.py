import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import init

class ResnetDepthMeanVarianceNetPlus(nn.Module):
    def __init__(self, input_channels, hidden_dim=256):
        super(ResnetDepthMeanVarianceNetPlus, self).__init__()
        # 使用 ResNet18 作为特征提取器
        self.feature_extractor = models.resnet18(pretrained=False)
        # 替换第一层卷积，适配输入通道数
        self.feature_extractor.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # 移除最后的全连接层和池化层
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc_mean_1 = nn.Linear(512+4, hidden_dim)  # ResNet18 最后一层通道数为 512
        self.fc_mean_2 = nn.Linear(hidden_dim, 1)
        self.fc_variance_1 = nn.Linear(512+4, hidden_dim)
        self.fc_variance_2 = nn.Linear(hidden_dim, 1)

    def initialize_weights(self):
        # 使用 Kaiming 初始化全连接层的权重
        init.kaiming_normal_(self.fc_mean_1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc_mean_2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc_variance_1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc_variance_2.weight, mode='fan_in', nonlinearity='relu')
        # 初始化偏置为 0
        init.constant_(self.fc_mean_1.bias, 0)
        init.constant_(self.fc_mean_2.bias, 0)
        init.constant_(self.fc_variance_1.bias, 0)
        init.constant_(self.fc_variance_2.bias, 0)
    
    def forward(self, x, camera):
        # 特征提取
        x = self.feature_extractor(x)
        # 全局平均池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        # 将 camera 特征拼接到 x 上
        x = torch.cat((x, camera), dim=1)
        # 输出均值和方差
        mean = self.fc_mean_1(x)
        mean = F.relu(mean)
        mean = self.fc_mean_2(mean)
        variance = self.fc_variance_1(x)
        variance = F.relu(variance)
        variance = self.fc_variance_2(variance)
        # 使用 Softplus 确保方差非负
        variance = F.softplus(variance)
        return mean, variance

if __name__ == "__main__":
    # 初始化模型
    input_channels = 8
    model = ResnetDepthMeanVarianceNetPlus(input_channels)

    # 加载本地 ResNet18 权重文件
    weight_path = "./base_ckpt_dir/dmvn_resnet/dmvn_5.pth"
    resnet_state_dict = torch.load(weight_path)

    # 将 ResNet18 的权重加载到 feature_extractor 中
    # 注意：需要过滤掉最后一层全连接层的权重
    resnet_state_dict = {k: v for k, v in resnet_state_dict.items() if k.startswith("feature_extractor")}
    print("len(resnet_state_dict):", len(resnet_state_dict))
    model.feature_extractor.load_state_dict(resnet_state_dict, strict=False)

    # 初始化全连接层权重
    model.initialize_weights()

    # 保存整个 ResnetDepthMeanVarianceNet 的权重
    save_path = "./base_ckpt_dir/dmvn_resnet/dmvnp_origin.pth"
    torch.save(model.state_dict(), save_path)
    print("len(model.state_dict()):", len(model.state_dict()))
    print(f"Model weights saved to {save_path}")