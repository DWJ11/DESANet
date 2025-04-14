import torch


# def make_optimizer(cfg, model, center_criterion):
#     params = []
#     for key, value in model.named_parameters():
#         # print(key)
#         if not value.requires_grad:
#             continue 
#         if 'classifier3' in key:
#             continue 
#         # if 'base'  in key or 'CRM'  in key or 'classifier.weight'  in key or 'bottleneck.weight'  in key or 'classifier' in key or 'bottleneck' in key:
            
#         # if  'base.blocks.0' in key  or 'CRM' in key or 'base.blocks.1' in key or 'base.blocks.2' in key or 'base.blocks.3' in key or 'base.blocks.4' in key or 'base.blocks.5' in key or 'base.blocks.6' in key or 'base.cls_token' in key or 'base.pos_embed' in key or 'base.sie_embed' in key:    
#         #     # print("key",key)
#         #     continue
#         # print("key",key)
#         lr = cfg.SOLVER.BASE_LR
#         weight_decay = cfg.SOLVER.WEIGHT_DECAY
#         if "bias" in key:
#             lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
#             weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
#         if cfg.SOLVER.LARGE_FC_LR:
#             if "classifier" in key or "arcface" in key:
#                 lr = cfg.SOLVER.BASE_LR * 2
#                 print('Using two times learning rate for fc ')
        
#         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

#     if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
#         optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
#     elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
#         optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
#     else:
#         optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
#     optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

#     # params2 = []
#     # for key, value in model.named_parameters():

#     #     if not value.requires_grad:
#     #         continue 
#     #     # if 'base' in key or 'CRM' in key or 'classifier.weight' in key or 'bottleneck.weight'  in key or 'classifier' in key or 'bottleneck' in key:
#     #     #     continue 
#     #     if 'classifier3' in key:
#     #     # if key == 'CRM.disNI.fc.weight' or key == 'CRM.disNI.fc.bias' or key == 'CRM.disTI.fc.weight' or key == 'CRM.disTI.fc.bias' or key == 'CRM.disRGB.fc.weight' or key == 'CRM.disRGB.fc.bias':
#     #         lr = 1e-7
#     #         weight_decay = cfg.SOLVER.WEIGHT_DECAY
#     #         params2 += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
#     #         print('key2',key)
#     # optimizer_center = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params2, momentum=cfg.SOLVER.MOMENTUM)

#     return optimizer, optimizer_center
#     # return optimizer, optimizer_center, discriminator_optimizer




#######################################################################################################
########best
def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        # print(key)
        if not value.requires_grad:
            continue 
        if 'classifier3' in key:
            continue 
        # if 'base'  in key or 'CRM'  in key or 'classifier.weight'  in key or 'bottleneck.weight'  in key or 'classifier' in key or 'bottleneck' in key:
            
        # if  'base.blocks.0' in key  or 'CRM' in key or 'base.blocks.1' in key or 'base.blocks.2' in key or 'base.blocks.3' in key or 'base.blocks.4' in key or 'base.blocks.5' in key or 'base.blocks.6' in key or 'base.cls_token' in key or 'base.pos_embed' in key or 'base.sie_embed' in key:    
        #     # print("key",key)
        #     continue
        # print("key",key)
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    # optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    params2 = []
    for key, value in model.named_parameters():

        if not value.requires_grad:
            continue 
        # if 'base' in key or 'CRM' in key or 'classifier.weight' in key or 'bottleneck.weight'  in key or 'classifier' in key or 'bottleneck' in key:
        #     continue 
        if 'classifier3' in key:
        # if key == 'CRM.disNI.fc.weight' or key == 'CRM.disNI.fc.bias' or key == 'CRM.disTI.fc.weight' or key == 'CRM.disTI.fc.bias' or key == 'CRM.disRGB.fc.weight' or key == 'CRM.disRGB.fc.bias':
            lr = 1e-7
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            params2 += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            print('key2',key)
    optimizer_center = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params2, momentum=cfg.SOLVER.MOMENTUM)

    return optimizer, optimizer_center
    # return optimizer, optimizer_center, discriminator_optimizer