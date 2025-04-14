import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable

# import ot
from scipy.spatial.distance import cdist
# class SOTLoss(nn.Module):
#     def __init__(self,p,metric='euclidean'):
#         super(SOTLoss, self).__init__()
#         self.p = p
#         self.metric = metric
#         print(f"using sot,{self.metric}")
#     def forward(self, a, b):
#         """ Args:
#             a, b: samples sets drawn from α,β respectively
#             p: the coefficient in the OT cost (i.e., the p in p-Wasserstein)
#             metric: the metric to compute cost matrix, 'euclidean' or 'cosine'
#         """
#         # cost matrix
#         # 
#         # if self.metric=="euclidean":
#         #     M = ot.dist(a, b, metric=self.metric)
#         # elif self.metric=="cosine":
#         #     # M = F.cosine_similarity(a, b)
#         #     M = cdist(a.cpu().numpy(),b.cpu().numpy(),metric=self.metric)
#         # else:
#         #     raise NotImplementedError
#         M = cdist(a.detach().cpu().numpy(),b.detach().cpu().numpy(),metric=self.metric)
#         M = torch.tensor(pow(M, self.p)).cuda()

#         # uniform distribution assumption
#         alpha = torch.tensor(ot.unif(len(a))).cuda()
#         beta = torch.tensor(ot.unif(len(b))).cuda()
        
#         # p-Wasserstein Distance
#         pW = ot.emd2(alpha, beta, M, numItermax=100000)
#         # pW = ot.sinkhorn2(alpha, beta, M, reg=.5, numItermax=1000, method='sinkhorn')
#         pW = pow(pW, 1/self.p)
    
#         return pW

# class SDLoss(nn.Module):
#     def __init__(self,p,metric='euclidean'):
#         super(SOTLoss, self).__init__()
#         self.p = p
#         self.metric = metric
#         print(f"using sot,{self.metric}")
#     def forward(self, a, b):
#         """ Args:
#             a, b: samples sets drawn from α,β respectively
#             p: the coefficient in the OT cost (i.e., the p in p-Wasserstein)
#             metric: the metric to compute cost matrix, 'euclidean' or 'cosine'
#         """
#         M = cdist(a.detach().cpu().numpy(),b.detach().cpu().numpy(),metric=self.metric)
#         M = torch.tensor(pow(M, self.p)).cuda()

#         # uniform distribution assumption
#         alpha = torch.tensor(ot.unif(len(a))).cuda()
#         beta = torch.tensor(ot.unif(len(b))).cuda()
        
#         # p-Wasserstein Distance
#         pW = ot.emd2(alpha, beta, M, numItermax=100000)
#         pW = pow(pW, 1/self.p)  
#         return pW
# loss_dis = 0.7*criterion_sot(rfeat, ifeat)+ 0.3*criterion_sot(ifeat, rfeat)

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss
class KLDivLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super(KLDivLoss, self).__init__()
        print(f'using KDloss,t={temperature}')
        self.temperature = temperature

    def forward(self, z_s, z_t, **kwargs):
        log_pred_student = F.log_softmax(z_s / self.temperature, dim=1)
        pred_teacher = F.softmax(z_t / self.temperature, dim=1)
        kd_loss = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        kd_loss *= self.temperature ** 2
        return kd_loss
class JSDLoss(nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()
        print(f'using JSDloss')
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p_output, q_output, **kwargs):
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
        log_mean_output = ((p_output + q_output )/2).log()
        return (self.KLDivLoss(log_mean_output, p_output) + self.KLDivLoss(log_mean_output, q_output))/2
        
    
class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

# class MMDLoss(nn.Module):
#     def __init__(self, kernel=RBF()):
#         super().__init__()
#         self.kernel = kernel

#     def forward(self, X, Y):
#         K = self.kernel(torch.vstack([X, Y]))
#         X_size = X.shape[0]
#         XX = K[:X_size, :X_size].mean()
#         XY = K[:X_size, X_size:].mean()
#         YY = K[X_size:, X_size:].mean()
#         return XX - 2 * XY + YY
    
class MMD_loss(nn.Module):

    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
 
class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

class CenterTripletLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni,label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num*2, 0)
        center = []
        for i in range(label_num*2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, batch_size, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, input, target):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = self.batch_size
        input1 = input.narrow(0,0,n)
        input2 = input.narrow(0,n,n)
        
        # Compute pairwise distance, replace by the official when merged
        dist = pdist_torch(input1, input2)
        
        # For each anchor, find the hardest positive and negative
        # mask = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i,i].unsqueeze(0))
            dist_an.append(dist[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct*2
        
class BiTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.suffix
    """
    def __init__(self, batch_size, margin=0.5):
        super(BiTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, input, target):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = self.batch_size
        input1 = input.narrow(0,0,n)
        input2 = input.narrow(0,n,n)
        
        # Compute pairwise distance, replace by the official when merged
        dist = pdist_torch(input1, input2)
        
        # For each anchor, find the hardest positive and negative
        # mask = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i,i].unsqueeze(0))
            dist_an.append(dist[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss1 = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct1  =  torch.ge(dist_an, dist_ap).sum().item() 
        
        # Compute pairwise distance, replace by the official when merged
        dist2 = pdist_torch(input2, input1)
        
        # For each anchor, find the hardest positive and negative
        dist_ap2, dist_an2 = [], []
        for i in range(n):
            dist_ap2.append(dist2[i,i].unsqueeze(0))
            dist_an2.append(dist2[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap2 = torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)
        
        # Compute ranking hinge loss
        y2 = torch.ones_like(dist_an2)
        # loss2 = self.ranking_loss(dist_an2, dist_ap2, y2)
        
        loss2 = torch.sum(torch.nn.functional.relu(dist_ap2 + self.margin - dist_an2))
        
        # compute accuracy
        correct2  =  torch.ge(dist_an2, dist_ap2).sum().item()
        
        loss = torch.add(loss1, loss2)
        return loss, correct1 + correct2
        
        
class BDTRLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.suffix
    """
    def __init__(self, batch_size, margin=0.5):
        super(BDTRLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, inputs, targets):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        correct  =  torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx



import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):

    def __init__(self, batch_size, temperature=0.1, use_cosine_similarity=True, alpha_weight=0.75):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim = 1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs,
                    norm=True,
                    weights=1.0):
        temperature = self.temperature
        alpha = self.alpha_weight

        """
        Pytorch implementation of the loss  SimCRL function by googleresearch: https://github.com/google-research/simclr
        @article{chen2020simple,
                title={A Simple Framework for Contrastive Learning of Visual Representations},
                author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2002.05709},
                year={2020}
                }
        @article{chen2020big,
                title={Big Self-Supervised Models are Strong Semi-Supervised Learners},
                author={Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2006.10029},
                year={2020}
                }
        """

        LARGE_NUM = 1e9
        """Compute loss for model.
        Args:
        hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.
        tpu_context: context information for tpu.
        weights: a weighting number or vector.
        Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
        """
        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
            
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        labels = labels.cuda()
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        
        """
        Different from Image-Image contrastive learning
        In the case of Image-Text contrastive learning we do not compute the similarity function between the Image-Image and Text-Text pairs  
        """
        # logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,0, 1)) / temperature
        # logits_aa = logits_aa - masks * LARGE_NUM
        # logits_bb = torch.matmul(hidden2,  torch.transpose(hidden2_large,0, 1)) / temperature
        # logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large,0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large,0, 1)) / temperature

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)

        return alpha*loss_a + (1-alpha)*loss_b


from torch.nn.modules.loss import _Loss
# import prototype.linklink as link

# class NT_Xent_gather(_Loss):
#     r"""The normalized temperature-scaled cross entropy loss, based on
#     `"A Simple Framework for Contrastive Learning of Visual Representations" <https://arxiv.org/abs/2002.05709>`_
#     """

#     def __init__(self, batch_size, temperature=0.1):
#         super(NT_Xent_gather, self).__init__()
#         self.batch_size = batch_size
#         self.temperature = temperature
#         self.mask_positive = None
#         self.mask_negative = None

#         self.criterion = nn.CrossEntropyLoss(reduction='sum')
#         self.similarity_f = nn.CosineSimilarity(dim=2)

#     def forward(self, z_i, z_ib, z_j, z_jb, temperature=None):  # z_ib, z_jb: in-batch
#         bs = z_i.shape[0]
#         assert bs == self.batch_size
#         l_bs = z_ib.shape[0]

#         if temperature is None:
#             temperature = self.temperature

#         p0 = torch.cat((z_i, z_j), dim=0)
#         p1 = torch.cat((z_ib, z_jb), dim=0)
#         sim = self.similarity_f(p0.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

#         if self.mask_positive is None:
#             ids = torch.arange(0, bs, dtype=torch.long).to(z_i.device)
#             labels = link.get_rank() * bs + torch.arange(0, bs, dtype=torch.long).to(z_i.device)
#             # positive samples
#             self.mask_positive = torch.zeros([bs*2, l_bs*2]).bool()
#             self.mask_positive[ids+bs, labels] = 1
#             self.mask_positive[ids, labels+l_bs] = 1
#             # negative samples
#             self.mask_negative = torch.ones([bs*2, l_bs*2]).bool()
#             self.mask_negative[ids, labels] = 0
#             self.mask_negative[ids+bs, labels] = 0
#             self.mask_negative[ids, labels+l_bs] = 0
#             self.mask_negative[ids+bs, labels+l_bs] = 0

#         positive_samples = sim[self.mask_positive].reshape(self.batch_size * 2, -1)
#         negative_samples = sim[self.mask_negative].reshape(self.batch_size * 2, -1)

#         labels = torch.zeros(self.batch_size * 2).to(z_i.device).long()
#         logits = torch.cat((positive_samples, negative_samples), dim=1)
#         loss = self.criterion(logits, labels)
#         loss /= 2 * self.batch_size
#         # import ipdb
#         # ipdb.set_trace()
#         return loss
class NT_Xent(_Loss):
    r"""The normalized temperature-scaled cross entropy loss, based on
    `"A Simple Framework for Contrastive Learning of Visual Representations" <https://arxiv.org/abs/2002.05709>`_
    """

    def __init__(self, batch_size, temperature=0.5):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            self.batch_size * 2, 1
        )
        negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(z_i.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss