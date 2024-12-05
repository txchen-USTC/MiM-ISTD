Official pytorch code of our TGRS 2024 paper "MiM-ISTD: Mamba-in-Mamba for Efficient Infrared Small Target Detection". 

[https://ieeexplore.ieee.org/abstract/document/10740056]

## News

* 24-11-01. Our paper get published in IEEE Transactions on Geoscience and Remote Sensing [IF=7.5].
  
* 24-03-15. We have corrected some errors and updated the whole network structure code of our MiM-ISTD. Feel free to use it, especially to more other tasks!

* 24-03-08. Our paper has been released on arXiv.

## A Quick Overview

![image](https://github.com/txchen-USTC/MiM-ISTD/blob/main/asset/overview.jpg)

## Efficiency Advantages

![image](https://github.com/txchen-USTC/MiM-ISTD/blob/main/asset/efficiency.jpg)

## Detailed structure of our Mamba-in-Mamba design

![image](https://github.com/txchen-USTC/MiM-ISTD/blob/main/asset/structure.jpg)

## Performance Comparison

![image](https://github.com/txchen-USTC/MiM-ISTD/blob/main/asset/performance.jpg)

## Required Environments

```
conda create -n mim python=3.8
conda activate mim
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

The .whl files of causal_conv1d and mamba_ssm could be found here. {[Baidu](https://pan.baidu.com/s/1Uza8g1pkVcbXG1F-2tB0xQ?pwd=p3h9)}

## Checkpoint

A newly retrained MiM checkpoint that maintains relatively high accuracy (around 80% IoU) on the SIRST dataset is available at Baidu Disk: {[Baidu](https://pan.baidu.com/s/1fyxlTmKG4HMvG07jrvoZIQ)}, extraction code: 3915. 

## Citation

Please cite our paper if you find the repository helpful.
```
@article{chen2024mim,
  title={Mim-istd: Mamba-in-mamba for efficient infrared small target detection},
  author={Chen, Tianxiang and Ye, Zi and Tan, Zhentao and Gong, Tao and Wu, Yue and Chu, Qi and Liu, Bin and Yu, Nenghai and Ye, Jieping},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
