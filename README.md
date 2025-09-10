# Project Name

### 1. 环境配置

```bash
conda create -m myhoi python=3.12 -y

conda init && source ~/.bashrc

conda activate myhoi

# Python 3.11/3.12 + PyTorch 2.4 + CUDA 12.1（最常用稳定组合）
# Python 3.11/3.12 + PyTorch 2.4 + CUDA 11.8（如需兼容旧显卡）
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 使用清华源镜像安装pytorch3d=0.7.8
conda install  pytorch3d-0.7.8-py312_cu121_pyt241.tar.bz2

pip install git+https://github.com/openai/CLIP.git

pip install numpy==2.0.1
pip install pyglet==1.5.31.
pip install chumpy==0.70

```


