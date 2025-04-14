import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', dim: int = 1, eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.dim = dim
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 计算余弦相似度 [batch_size, ...]
        cos_sim = F.cosine_similarity(input, target, dim=self.dim, eps=self.eps)
        # 转换为损失值: 1 - 相似度
        loss = 1.0 - cos_sim
        
        # 应用reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class FeatureKLLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 输入需要是log概率，目标需要是概率
        log_input = F.log_softmax(input, dim=1)
        target_prob = F.softmax(target, dim=1)
        
        # 计算KL散度 (batch_size,)
        kl_div = F.kl_div(log_input, target_prob, reduction='none').sum(dim=1)
        
        # 应用reduction
        if self.reduction == 'mean':
            return kl_div.mean()
        elif self.reduction == 'sum':
            return kl_div.sum()
        else:  # 'none'
            return kl_div

class ContrastiveFeatureLoss(nn.Module):
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = input.size(0)
        
        # 计算正样本对距离 (anchor-positive)
        pos_dist = F.pairwise_distance(input, target, p=2)  # [batch_size]
        
        # 生成负样本对 (随机打乱target作为negative)
        neg_indices = torch.randperm(batch_size, device=input.device)
        neg_target = target[neg_indices]
        neg_dist = F.pairwise_distance(input, neg_target, p=2)  # [batch_size]
        
        # 计算对比损失
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        
        # 应用reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses

"""
使用示例:
cos_loss = CosineSimilarityLoss(reduction='mean')
kl_loss = FeatureKLLoss(reduction='sum')
cont_loss = ContrastiveFeatureLoss(margin=0.5, reduction='none')

input = torch.randn(32, 256)  # 假设特征维度256
target = torch.randn(32, 256)

print(cos_loss(input, target).shape)  # scalar
print(kl_loss(input, target).shape)   # scalar 
print(cont_loss(input, target).shape) # [32]
"""
