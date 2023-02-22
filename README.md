# Gamma-Spec
Gamma-Spec is a spectral graph neural network learning library based on PyTorch. This open-source framework integrates a variety of spectral neural network models, builds a unified underlying platform, encapsulates the data processing process, unifies the input and output interfaces, abstracts the key operators of the algorithm, and visualizes the filters. And we evaluate the performance of the models under a unified evaluation standard.

<p align="center">
    <img src="https://raw.githubusercontent.com/liuyang-tian/Spectral-GNN-Library/master/Overall%20Architecture.png" width="600">
    <br>
    <b>Figure</b>: Gamma-Spec Overall Architecture
</p>

## Supported Models
<table>
    <tr>
        <td><a href="https://arxiv.org/abs/2104.12840">AdaGNN</a></td>
        <td> <a href="https://arxiv.org/abs/2112.07160">Correlation-free</a></td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2112.04575">AKGNN</a></td>
        <td><a href="https://arxiv.org/abs/1909.12038">DSGC</a></td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1901.01343">ARMA</a></td>
        <td><a href="https://arxiv.org/abs/2101.00797">FAGCN</a></td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2106.10994">BernNet</a></td>
        <td><a href="https://arxiv.org/abs/2006.07988">GPR-GNN</a></td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1705.07664">CayleyNet</a></td>
        <td><a href="https://arxiv.org/abs/2205.11172">JacobiNet</a></td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1606.09375">ChebyNet</a></td>
        <td><a href="https://arxiv.org/abs/1901.01484">LanczosNet</a></td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2202.03580">ChebyNetII</a></td>
        <td><a href="https://arxiv.org/abs/1312.6203">SpectralNet</a></td>
    </tr>
</table>


## Setup

**1. Python Environment**
```bash
conda create -n sgl python=3.9.12
source activate sgl
```

**2. Install Backend and Libraries**
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pyg-lib torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster torch-spline-conv torch-geometric==1.7.2 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install --pre dgl -f https://data.dgl.ai/wheels/cu113/repo.html
pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install matplotlib==3.5.1 easydict==1.10
```

## Run the Examples
```bash
cd examples
python adagnn_trainer.py
```

## Set Parameters
You can set parameters by modifying the files in `config` folder.
