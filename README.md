# DeColorful-Net:

**Close Imitation of Expert Retouching for Black-and-White Photography**<br>


## Introduction

__DeColorful-Net.__ We propose a DeColorfulNet, which is based on a DML framework with a hierarchical
proxy-based loss and hierarchical bilateral grid network to mimic the experts’ retouching scheme
## Prerequisites

- Python >= 3.6
- PyTorch >= 1.0
- NVIDIA GPU + CUDA cuDNN

## Dataset
Since our dataset is the first aesthetic decolorization dataset which contains three different set retouched by experts,\
there is no dataset that can be directly used to train our network.\
We are ready for release our dataset.

## Getting Started

Our DeColorful-Net consists of two steps.
1. Proxy-Generation
2. Decolorization 

Before train your model you should change some options with your settings which are listed in the form of json file.\
ex)\
dataset_path : path to your dataset\
tb_log: turn on/off to save log files\
tb_freq: how often to save log files\
etc.

You can see json file with below command 
```
cd ./workspace/Expert/Multi_Encoder
```
### Installation


- Install python requirements:

```
pip install -r requirements.txt
```

### Training Phase 1

```commandline
python train_step1.py --ws Expert --exp Multi_Encoder --args json 
```

ws: workspace\
exp: experience space\
args: which format to use options


### Training Phase 2

Make sure to change directory of pretrained model from training phase 1.

```commandline
python train_step2.py --ws Expert --exp Multi_Encoder --args json
```

### Inference

Make sure to change directory of pretrained model from training phase 1 & 2.\
Also you should change style index corresponding to your preferred style in json file.

```commandline
python test.py --ws Expert --exp Multi_Encoder --args json
```


Our example of dataset and Userstudy results are also provided in pdf format. \
Since Supplementary file is up to 100MB, we randomly select 40 images from each expert's set. \
Also, to show more results of user study, we compressed results videos and pdf to be under limit.
