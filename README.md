![Python >=3.5](https://img.shields.io/badge/Python->=3.6-blue.svg)
![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.6-yellow.svg)

# Multi-Level Contrastive Learning for Unsupervised Vessel Re-Identification


## Requirements

### Installation

```shell
cd MCL
python setup.py develop
```


## Training

We utilize 4 GTX-1080TI GPUs for training.

**examples:**

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/mcl_train.py -b 128 -a resnet50 -d vesselreid --iters 200 --eps 0.4 --num-instances 4
```


## Evaluation

To evaluate the model, run:
```shell
CUDA_VISIBLE_DEVICES=0 \
python examples/test.py \
  -d vesselreid --resume path/to/model_best.pth.tar
```


# Acknowledgements

Thanks to Alibaba for opening source of excellent works  [ClusterContrast](https://github.com/alibaba/cluster-contrast-reid). 
