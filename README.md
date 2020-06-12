# Inductive Graph Neural Networks for Spatiotemporal Kriging (IGNNK) -- Code

This is the code corresponding to the experiments conducted for the work "Inductive Graph Neural Networks for Spatiotemporal Kriging"
(Yuankai Wu, Dingyi Zhuang, Aurelie Labbe and Lijun Sun).

## Tasks

<img src="https://github.com/Kaimaoge/IGNNK/blob/master/fig/Fig1new2-1.png" width="800">
<img src="https://github.com/Kaimaoge/IGNNK/blob/master/fig/Fig2new2-1.png" width="800">

The goal of spatiotemporal kriging is to perform signal interpolation for unsampled locations given the observed signals from sampled locations during the same period. We first randomly select a subset of nodes from all available sensors and create a corresponding subgraph. We mask some of them as missing and train the GNN to reconstruct the full signals of all nodes (including both the observed and the masked nodes) on the subgraph.

## Datasets

The datasets manipulated in this code can be downloaded on the following locations:
- the METR-LA traffic data: https://github.com/liyaguang/DCRNN;
- the NREL solar energy: https://www.nrel.gov/grid/solar-power-data.html
- the USHCN weather condition: https://www.ncdc.noaa.gov/ushcn/introduction
- the SeData traffic data: https://github.com/zhiyongc/Seattle-Loop-Data
- the PeMS traffic data: https://github.com/liyaguang/DCRNN

## Files

- `utils.py` file: preprocess datasets;
- `basic_structure.py` file: pytorch implementation of basic graph neural network structure
- `IGNNK_D_METR_LA.ipynb` file: a training example on METR_LA dataset

### Basic GNNs implementation (basic_structure.py)

#### Graph convolutional networks - K_GCN in basic_structure.py

- Kipf, Thomas N., and Max Welling. ("Semi-Supervised Classification with Graph Convolutional Networks.")[https://arxiv.org/pdf/1609.02907.pdf] (ICLR 2016).

#### Chebynet - C_GCN

- Micha ̈el Defferrard, Xavier Bresson, and Pierre Vandergheynst. ("Convolutional neural networks ongraphs with fast localized spectral filtering.")[http://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf] (NIPS 2016).

#### Diffusion convolutional networks - D_GCN

- Li, Y., Yu, R., Shahabi, C., & Liu, Y. ("Diffusion convolutional recurrent neural network: Data-driven traffic forecasting.")[https://arxiv.org/pdf/1707.01926.pdf] (ICLR 2017).

Our IGNNK structure is based on the diffusion convolutional networks, one can always build his own structure using those basic building blocks. We will continue implementing more GNN structures that are suitable for kriging tasks.

#### Graph attention networks - GAT

- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. ("Graph attention networks.")[https://arxiv.org/pdf/1710.10903.pdf] (NIPS 2017).


### Training on the METR_LA datasets

python ......


