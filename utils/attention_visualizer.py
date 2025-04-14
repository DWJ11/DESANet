
import torch
import torch.nn.functional as F
import cv2
from matplotlib import pyplot as plt
import numpy as np


class MultimodalGradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.model.register_hooks()  # 注册修正后的钩子
        
    def __call__(self, input_dict):
        # 前向传播
        with torch.enable_grad():
            outputs = self.model(input_dict)
        
        # 获取目标类别
        target = outputs.argmax(dim=1)
        outputs[:, target].sum().backward()
        
        # 生成各模态热力图
        heatmaps = {}
        for mod in ['RGB', 'NI', 'TI']:
            features = self.model.attention_features[mod]
            grads = features.grad.mean(dim=1, keepdim=True)
            
            # 计算加权特征
            cam = (grads * features).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, input_dict[mod].shape[-2:], mode='bilinear')
            
            # 归一化
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            heatmaps[mod] = cam.detach().cpu().numpy()
            
        return heatmaps


def visualize_multimodal_attention(input_dict, heatmaps, save_path=None):
    """可视化三模态输入及其热力图"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    
    # 颜色映射定义
    cmaps = {
        'RGB': cv2.COLORMAP_JET,
        'NI': cv2.COLORMAP_VIRIDIS,
        'TI': cv2.COLORMAP_INFERNO
    }
    
    for row, mod in enumerate(['RGB', 'NI', 'TI']):
        # 原始图像
        img = input_dict[mod][0].permute(1,2,0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        axes[row,0].imshow(img)
        axes[row,0].set_title(f'Raw {mod} Image', fontsize=12)
        axes[row,0].axis('off')
        
        # 热力图叠加
        heatmap = heatmaps[mod][0][0]
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cmaps[mod])
        
        # 融合显示
        superimposed = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
        axes[row,1].imshow(superimposed)
        axes[row,1].set_title(f'{mod} Attention Map', fontsize=12)
        axes[row,1].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
