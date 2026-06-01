<div align="left">

# Escaping Modal Interactions: An Efficient DESANet for Multi-Modal Object Re-identification (TIP 2025)

</div>

<p align="left">
Official PyTorch Implementation of Escaping Modal Interactions: An Efficient DESANet for Multi-Modal Object Re-identification
</p>

<p align="left">
  <a href="https://ieeexplore.ieee.org/document/11104996">📄 Paper</a>
</p>

---

# Abstract

Multi-modal object Re-ID aims to leverage the complementary information provided by multiple modalities to overcome challenging conditions and achieve high-quality object matching. However, existing multi-modal methods typically rely on various modality interaction modules for information fusion, which can reduce the efficiency of real-time monitoring systems. Additionally, practical challenges such as low-quality multi-modal data or missing modalities further complicate the application of object Re-ID. To address these issues, we propose the Complementary Data Enhancement and Modal-Aware Soft Alignment Network (DESANet), which is designed to be independent of interactive networks and adaptable to scenarios with missing modalities. This approach ensures a simple-yet-effective, and efficient multi-modal object Re-ID. DESANet consists of three key components: Firstly, the Dual-Color Space Data Enhancement (DCDE) module, which enhances multi-modal data by performing patch rotation in the RGB space and improving image quality in the HSV space. Secondly, the Salient Feature ReConstruction (SFRC) module, which addresses the issue of missing modalities by reconstructing features from one modality using the other two. Thirdly, the Modal-Aware Soft Alignment (MASA) module, which integrates multi-source data to avoid the blind fusion of features and prevents the propagation of noise from reconstructed modalities. Our approach achieves state-of-the-art performances on both person and vehicle datasets. Source code is available at https://github.com/DWJ11/DESANet.
---


# Framework Overview

<p align="center">
  <img src="https://github.com/DWJ11/DESANet/blob/main/figures/main.png" width="95%">
</p>

---



# Experimental Results

## Person

<p align="center">
  <img src="https://github.com/DWJ11/DESANet/blob/main/figures/person.png" width="90%">
</p>


## Vehicle

<p align="center">
  <img src="https://github.com/DWJ11/DESANet/blob/main/figures/vehicle.png" width="90%">
</p>

### Models
<a href="https://pan.quark.cn/s/84978254c6b3">Download</a> 




---

# Visualization Results


## T-SNE results

<p align="center">
  <img src="https://github.com/DWJ11/DESANet/blob/main/figures/tsne.png" width="95%">
</p>

---

# Citation

If you find this project useful for your research, please consider citing:

```bibtex
@article{dong2025escaping,
title={Escaping Modal Interactions: An Efficient DESANet for Multi-Modal Object Re-Identification},
author={Dong, Wenjiao and Yang, Xi and Cheng, De and Wang, Nannan and Gao, Xinbo},
journal={IEEE Transactions on Image Processing},
year={2025},
publisher={IEEE}
}

```

---

<div align="center">

</div>

</div>
