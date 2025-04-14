import torch
import torch.nn as nn
# from modeling.backbones.mamba import vvim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2,vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
# from modeling.backbones.mamba import 
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.fusion_part.CRM import CRM
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
# import ot
import numpy as np
from .OTKE import OTKernel
from torch.nn import functional as F
import random
from layers.contrastive_losses import Triple_Contrastive_Loss
from torch.nn import L1Loss, MSELoss




def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # 仅处理包含weight参数的层
    if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    # 对BatchNorm层特殊处理（可选）
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)



def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE
        if 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 512
        if 'edge' in cfg.MODEL.TRANSFORMER_TYPE or cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224':
            self.in_planes = 384
        if '14' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 384
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        # No view
        view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        num_classes=num_classes,
                                                        camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        cash_x = self.base(x, cam_label=cam_label, view_label=view_label)
        # print(cash_x.shape)
        global_feat = cash_x[-1][:, 0]
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cash_x, cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # cls_score = self.classifier(feat)
                return cash_x,feat
            else:
                # cls_score = self.classifier(feat)
                return cash_x, global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class TOPReID(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(TOPReID, self).__init__()
        self.NI = build_transformer(num_classes, cfg, camera_num, view_num, factory)
        self.TI = build_transformer(num_classes, cfg, camera_num, view_num, factory)
        self.RGB = build_transformer(num_classes, cfg, camera_num, view_num, factory)

        self.num_classes = num_classes
        self.cfg = cfg
        self.camera = camera_num
        self.view = view_num
        self.num_head = 12
        self.mix_dim = 768
        if 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.mix_dim = 512
            self.num_head = 8
        if 'deit' in cfg.MODEL.TRANSFORMER_TYPE or cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224':
            self.mix_dim = 384
        
        self.re = cfg.MODEL.RE
        if self.re:
            self.CRM = CRM(dim=self.mix_dim, num_heads=self.num_head, miss=cfg.TEST.MISS,
                           depth=cfg.MODEL.RE_LAYER)
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.layer = cfg.MODEL.LAYER
        self.direct = cfg.MODEL.DIRECT
        self.scale = 300
        self.ratio = 3
        self.alpha = 1e-1
        self.gamma = 1
        self.data = cfg.DATASETS.NAMES


        if cfg.DATASETS.NAMES == 'RGBNT201' or cfg.DATASETS.NAMES == 'market_to_RGBNT201':
            self.beta = 0.1
        else:
            self.beta = 0.3
        if self.training:
            self.loss = Triple_Contrastive_Loss()
            self.classifier3 = nn.Linear(3 * self.mix_dim, 1)  ###correlation predictor
            self.classifier3.apply(weights_init_classifier)
            self.loss1 = L1Loss(margin=0.3)
                    
            # self.loss = MSELoss()
            self.classifier_ViT = nn.Linear(3 * self.mix_dim, self.num_classes, bias=False)
            self.classifier_ViT.apply(weights_init_classifier)
            self.bottleneck_ViT = nn.BatchNorm1d(3 * self.mix_dim)
            self.bottleneck_ViT.bias.requires_grad_(False)
            self.bottleneck_ViT.apply(weights_init_kaiming)

        self.miss = cfg.TEST.MISS


        
    def generate_trimodal_loss(self, x_l, x_a, x_v, label = True):
        
        batch = [i for i in range(x_l.shape[0])]
        index1 = np.random.choice(batch, x_l.shape[0] * (self.ratio+1), replace=True)
        index2 = np.random.choice(batch, x_l.shape[0] * (self.ratio+1), replace=True)
        index3 = np.where(index1 != index2)[0]
        index1 = index1[index3]
        index2 = index2[index3]
        index3 = np.random.choice(batch, len(index1), replace=True)

        index4 = np.where(index1 != index3)[0]
        index5 = np.where(index2 != index3)[0]
        index4 = set(index4).intersection(set(index5))
        index4 = list(index4)
        index1 = index1[index4]
        index2 = index2[index4]
        index3 = index3[index4]
        
        total_len = x_l.shape[0] * self.ratio
        if len(index1) > total_len:
            index1 = index1[:total_len]
            index2 = index2[:total_len]
            index3 = index3[:total_len]

 
        label_n  = torch.zeros((len(index1),1),).float().cuda()
        label_p = torch.ones((x_l.shape[0],1),).float().cuda()

        negative_pair1 = torch.cat([x_l[index1], x_v[index2], x_a[index3]], dim = -1)
        positive_pair1 = torch.cat([x_l, x_v, x_a], dim = -1)
        pair1 = torch.cat([negative_pair1, positive_pair1], dim=0)
        y1 = self.classifier3(pair1)
        label = torch.cat([label_n+1, label_p-1], dim=0)
        loss2 = self.loss1(y1, label)  
        return loss2


    

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')  
        own_state = self.state_dict()
        mismatch_log = []
        skipped_params = []

        for param_name in param_dict:

            target_name = param_name.replace('module.', '')
            
            if target_name not in own_state:
                skipped_params.append(param_name)

            if param_dict[param_name].shape != own_state[target_name].shape:
                mismatch_log.append(
                    f"跳过 {target_name} | 预训练形状: {tuple(param_dict[param_name].shape)} vs 当前形状: {tuple(own_state[target_name].shape)}"
                )
                continue

            own_state[target_name].copy_(param_dict[param_name])

        print(f"成功加载参数: {len(param_dict)-len(mismatch_log)-len(skipped_params)}/{len(param_dict)}")
        
        if mismatch_log:
            print("\n".join(["[不匹配参数]"] + mismatch_log))
        if skipped_params:
            print("\n".join(["[未找到参数]"] + skipped_params))




    def forward(self, x, cam_label=None, label=None, view_label=None):
        if self.training:
            RGB = x['RGB']
            # print(RGB.shape)
            NI = x['NI']
            TI = x['TI']
            NI_cash, NI_score, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_score, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_score, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)

            loss_con = self.loss(NI_global,TI_global,label)[0]
            loss_con += self.loss(TI_global,NI_global,label)[0]
            loss_con += self.loss(NI_global,RGB_global,label)[0]
            loss_con += self.loss(RGB_global,NI_global,label)[0]
            loss_con += self.loss(TI_global,RGB_global,label)[0]
            loss_con += self.loss(RGB_global,TI_global,label)[0]
            loss = self.generate_trimodal_loss(RGB_global, NI_global, TI_global,label) 


            
            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            ori_global = self.bottleneck_ViT(ori)
            ori_score = self.classifier_ViT(ori_global)
            if self.re:
                loss_re = self.CRM(RGB_cash[self.layer], NI_cash[self.layer], TI_cash[self.layer],flag = 1)
            if self.re:
                if self.direct:
                    return ori_score, ori_global, loss_re,loss+loss_con*self.beta
                else:
                    return ori_score, ori_global, RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global, loss_re,loss+loss_con*self.beta
            else:
                if self.direct:
                    return ori_score, ori_global, loss+loss_con*self.beta
                else:
                    return ori_score, ori_global, RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            # RGB = predata(RGB,self.data)
            NI_cash, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)
            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            if self.re:
                if self.miss == 'r':
                    RGB = self.CRM(ma=None, mb=NI_cash[self.layer], mc=TI_cash[self.layer],flag = 0,miss = 'r')
                    ori = torch.cat([RGB[:, 0,:], NI_global, TI_global], dim=-1)
                elif self.miss == "n":
                    NI = self.CRM(ma=RGB_cash[self.layer], mb=None, mc=TI_cash[self.layer],flag = 0,miss = 'n')
                    ori = torch.cat([RGB_global, NI[:, 0,:], TI_global], dim=-1)
                elif self.miss == 't':
                    TI = self.CRM(ma=RGB_cash[self.layer], mb=NI_cash[self.layer], mc=None,flag = 0,miss = 't')
                    ori = torch.cat([RGB_global, NI_global, TI[:, 0,:]], dim=-1)
                elif self.miss == 'rn':
                    RGB, NI = self.CRM(ma=None, mb=None, mc=TI_cash[self.layer],flag = 0,miss = 'rn')
                    ori = torch.cat([RGB[:,0,:], NI[:, 0,:], TI_global], dim=-1)
                elif self.miss == 'rt':
                    RGB, TI = self.CRM(ma=None, mb=NI_cash[self.layer], mc=None,flag = 0,miss = 'rt')
                    ori = torch.cat([RGB[:,0,:], NI_global, TI[:, 0,:]], dim=-1)
                elif self.miss == 'nt':
                    NI, TI = self.CRM(ma=RGB_cash[self.layer], mb=None, mc=None,flag = 0,miss = 'nt')
                    ori = torch.cat([RGB_global, NI[:,0,:], TI[:, 0,:]], dim=-1)
            ori_global = self.bottleneck_ViT(ori)
            return ori_global


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = TOPReID(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building TOPReID===========')
    return model
