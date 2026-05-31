## Get start

1. Train & Test
   ```python
   RGBNT201
Train：CUDA_VISIBLE_DEVICES=1 python train_net.py --config_file ./configs/RGBNT201/TOP-ReID.yml
Test(full)：CUDA_VISIBLE_DEVICES=0 python test_net.py --config_file ./configs/RGBNT201/TOP-ReID.yml --WEIGHT ./bestmodel/RGBNT201/TOP-ReID/TOPReIDbest.pth
Test(M(R))：CUDA_VISIBLE_DEVICES=0 python test_net.py --config_file ./configs/RGBNT201/TOP-ReID.yml --WEIGHT ./bestmodel/RGBNT201/TOP-ReID/TOPReIDbest.pth TEST.MISS r
Test(M(N))：CUDA_VISIBLE_DEVICES=0 python test_net.py --config_file ./configs/RGBNT201/TOP-ReID.yml --WEIGHT ./bestmodel/RGBNT201/TOP-ReID/TOPReIDbest.pth TEST.MISS n
Test(M(T))：CUDA_VISIBLE_DEVICES=0 python test_net.py --config_file ./configs/RGBNT201/TOP-ReID.yml --WEIGHT ./bestmodel/RGBNT201/TOP-ReID/TOPReIDbest.pth TEST.MISS t
Test(M(RN))：CUDA_VISIBLE_DEVICES=0 python test_net.py --config_file ./configs/RGBNT201/TOP-ReID.yml --WEIGHT ./bestmodel/RGBNT201/TOP-ReID/TOPReIDbest.pth TEST.MISS rn
Test(M(RT))：CUDA_VISIBLE_DEVICES=0 python test_net.py --config_file ./configs/RGBNT201/TOP-ReID.yml --WEIGHT ./bestmodel/RGBNT201/TOP-ReID/TOPReIDbest.pth TEST.MISS rt
Test(M(NT))：CUDA_VISIBLE_DEVICES=0 python test_net.py --config_file ./configs/RGBNT201/TOP-ReID.yml --WEIGHT ./bestmodel/RGBNT201/TOP-ReID/TOPReIDbest.pth TEST.MISS nt

MARKET_to_RGBNT
Train：CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file ./configs/market_to_RGBNT201/TOP-ReID.yml
Test：CUDA_VISIBLE_DEVICES=0 python test_net.py --config_file ./configs/market_to_RGBNT201/TOP-ReID.yml --WEIGHT ./bestmodel/MARKET_to_RGBNT201/TOP-ReID/TOPReIDbest.pth

RGBNT100
Train：CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file ./configs/RGBNT100/TOP-ReID.yml
Test：CUDA_VISIBLE_DEVICES=1 python test_net.py --config_file ./configs/RGBNT100/TOP-ReID.yml --WEIGHT ./bestmodel/RGBNT100/TOP-ReID/TOPReIDbest.pth

MSVR310
Train：CUDA_VISIBLE_DEVICES=0 python train_net.py --config_file ./configs/MSVR310/TOP-ReID.yml
Test：CUDA_VISIBLE_DEVICES=0 python test_net.py --config_file ./configs/MSVR310/TOP-ReID.yml --WEIGHT ./bestmodel/MSVR310/TOP-ReID/TOPReIDbest.pth
   ```

2. Bestpath
   Qurk Drive Link: https://pan.quark.cn/s/84978254c6b3


4.Reference：
   @article{dong2025escaping,
  title={Escaping Modal Interactions: An Efficient DESANet for Multi-Modal Object Re-Identification},
  author={Dong, Wenjiao and Yang, Xi and Cheng, De and Wang, Nannan and Gao, Xinbo},
  journal={IEEE Transactions on Image Processing},
  year={2025},
  publisher={IEEE}
}

