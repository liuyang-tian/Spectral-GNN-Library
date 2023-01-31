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
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt
```


## Run the Examples

```bash
cd examples
python adagnn_trainer.py
```

## Set Parameters
You can set parameters by modifying the files in `config` folder.
