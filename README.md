# Inductive Graph Neural Networks for Spatiotemporal Kriging (IGNNK)

This is the code corresponding to the experiments conducted for the AAAI 2021 paper "[Inductive Graph Neural Networks for Spatiotemporal Kriging](https://arxiv.org/abs/2006.07527)"
(Yuankai Wu, Dingyi Zhuang, Aurélie Labbe and Lijun Sun).

## Motivations

In many applications, placing sensors with fully spatial coverage may be impractical. Installation and maintenance costs of devices can also limit the number of sensors deployed in a network. A better kriging model can achieve higher estimation accuracy/reliability with less number of sensors, thus reducing the operation and maintenance cost of a sensor network. The kriging results can produce a fine-grained and high-resolution realization of spatiotemporal data, which can be used to enhance real-world applications such as travel time estimation and disaster evaluation.

A limitation with traditional methods is that they are essentially *transductive*, for new sensors/nodes introduced to the network, we cannot directly apply a previously trained model; instead, we have to retrain the full model even with only minor changes. Conversely, we develop an *Inductive* Graph Neural Network Kriging (IGNNK) model in this work. 

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

## Dependencies

* numpy
* pytorch
* matplotlib
* pandas
* scipy
* scikit-learn
* geopandas


## Files

- `utils.py` file: preprocess datasets;
- `basic_structure.py` file: pytorch implementation of basic graph neural network structure
- `IGNNK_D_METR_LA.ipynb` file: a training example on METR_LA dataset
- `IGNNK_U_Central_missing.ipynb` file: present the kriging of central US precipitation (USHCN weather condition)

### Basic GNNs implementation (basic_structure.py)

#### Graph convolutional networks - K_GCN in basic_structure.py

- Kipf, Thomas N., and Max Welling. ["Semi-Supervised Classification with Graph Convolutional Networks."](https://arxiv.org/pdf/1609.02907.pdf) (ICLR 2016).

#### Chebynet - C_GCN

- Micha ̈el Defferrard, Xavier Bresson, and Pierre Vandergheynst. ["Convolutional neural networks ongraphs with fast localized spectral filtering."](http://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf) (NIPS 2016).

#### Diffusion convolutional networks - D_GCN

- Li, Y., Yu, R., Shahabi, C., & Liu, Y. ["Diffusion convolutional recurrent neural network: Data-driven traffic forecasting."](https://arxiv.org/pdf/1707.01926.pdf) (ICLR 2017).

Our IGNNK structure is based on the diffusion convolutional networks, one can always builds his own structure using those basic building blocks. We will continue implementing more GNN structures that are suitable for kriging tasks.

#### Graph attention networks - GAT

- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. ["Graph attention networks."](https://arxiv.org/pdf/1710.10903.pdf) (NIPS 2017).


### Training on the METR_LA datasets
You can simply train IGNNK on METR-LA from command line by

`python IGNNK_train.py "metr" --n_o 150 --h 24 --n_m 50 --n_u 50 --max_iter 750`

for other datasets:
#### NREL
`python IGNNK_train.py "nrel" --n_o 100 --h 24 --n_m 30 --n_u 30 --max_iter 750`

#### USHCN
`python IGNNK_train.py "ushcn" --n_o 900 --h 6 --n_m 300 --n_u 300 --max_iter 750 --z 350`

#### SeData
`python IGNNK_train.py "sedata" --n_o 240 --h 24 --n_m 80 --n_u 80 --max_iter 750`


