
### 训练
行人
CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file ./configs/RGBNT201/TOP-ReID.yml
CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file ./configs/market_to_RGBNT201/TOP-ReID.yml
车辆
CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file ./configs/RGBNT100/TOP-ReID.yml
CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file ./configs/MSVR310/TOP-ReID.yml

### 测试
行人
CUDA_VISIBLE_DEVICES=1 python test_net.py --config_file ./configs/RGBNT201/TOP-ReID.yml --WEIGHT ..
CUDA_VISIBLE_DEVICES=0 python test_net.py --config_file ./configs/market_to_RGBNT201/TOP-ReID.yml --WEIGHT ..
车辆
CUDA_VISIBLE_DEVICES=0 python test_net.py --config_file ./configs/RGBNT100/TOP-ReID.yml --WEIGHT ..
CUDA_VISIBLE_DEVICES=0 python test_net.py --config_file ./configs/MSVR310/TOP-ReID.yml --WEIGHT ..

