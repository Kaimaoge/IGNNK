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

### Training on the METR_LA datasets

python ......


