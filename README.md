# Decoupling Supervised Learning for Information Regularization

We propose Decoupled Supervised Learning with Information Regularization (DeInfoReg), that decouples the gradients of different blocks to ensure that the gradients of different blocks do not interfere.
 
DeInfoReg enhances model performance and flexibility by designing new Local Loss and model structures. The new model structure endows the model with an Adaptive Inference Path, Dynamic Expanded Layers, and Dynamic Extended Layers with new features. 

The Local Loss function, employing three regularization methods. Those methods include: ensuring the invariance of the output embedding with true labels, maintaining the variance of output embedding within batch size, and using the covariance to reduce redundancy in the output embedding. This allows the model to better capture data features, enhancing performance.

The detailed content is written in the master's thesis, and the complete source code is published in [this project](https://drive.google.com/file/d/1Ngfl_mo9LsG8N2q_txE1Lej2ieUtlolT/view?usp=sharing).

---
## Environment

| Name | Version |
| --- | --- | 
| Python | `3.8.10` |
| CUDA | `12.0.1` | 
| PyTorch | `2.0.1+cu118` | 
| | |

---
## Setup

### Build an Environment

#### Using Docker
you can simulate the experiment using Docker by Nvidia ,with the following steps:
```bash
git clone https://github.com/ianzih/Decoupling-Supervised-Learning-for-Information-Regularization.git
docker run --ipc=host  --gpus all --name DeInfoReg -it -p 0.0.0.0:2486:2486 -v [DeInfoReg file path]:/workspace nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04
cd ./workspace
apt-get update -y && apt-get upgrade -y
apt-get install python3-pip -y
apt-get install wget -y
apt-get install git -y
apt-get install vim -y
apt-get install zip -y
apt-get install openssh-server -y && service ssh start && passwd root
echo "Port 2486" >> /etc/ssh/sshd_config && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && /etc/init.d/ssh restart
pip install networkx==3.0
pip install torch==2.1.0 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

If you prefer not to create the environment yourself, you can also use pre-prepared image directly from Docker Hub.
```bash
git clone https://github.com/ianzih/Decoupling-Supervised-Learning-for-Information-Regularization.git
docker run --ipc=host  --gpus all --name DeInfoReg -it -p 0.0.0.0:2486:2486 -v [DeInfoReg file path]:/workspace ianzih/deinforeg:24_06
cd ./workspace
```

### Dataset
#### Vision
* Tiny-imagenet-200:  Download [here](https://drive.google.com/file/d/1M6laujHUg1RJNzrTCnnw4nE6yuijiK1D/view?usp=drive_link). This zip file contains the tinyImageNet dataset processed in the PyTorch ImageFolder format.
  ```bash
  unzip tiny-imagenet-200.zip
  # Place the unzipped folder (`./tiny-imagenet-200`) in the root.
  ```

#### NLP
* IMDB: Please download the dataset from [here](https://drive.google.com/file/d/12aEMx9gfwMDZnzGjr3ztIjcCsN4UMW4M/view?usp=sharing).
  ```bash
  mkdir nlpdataset && cd ./nlpdataset
  # Put this file (`IMDB_Dataset.csv`) in the ./nlpdataset.
  ```

#### Download Word Embedding
* Glove
  ```bash
  # cd to the path of your project
  wget https://nlp.stanford.edu/data/glove.6B.zip --no-check-certificate
  unzip glove.6B.zip
  # "glove.6B.300d.txt" must be put in the root of the project
  ```
  
---
## Quick Start

### Vision

#### Example
```bash
# Command Example
python3 train_vision.py --model VGG_DeInfoReg --dataset cifar100 --epochs 200 --optimal LARS
```
#### Usage
```bash
python3 train_vision.py [Options]
```
#### Options
| Name | Default | Description |
| -------- | -------- | -------- |
|"--task"|"vision"|Task|
|"--model"|"VGG_DeInfoReg"|Model Name </br> ("VGG", "VGG_AL", "VGG_SCPL", "VGG_DeInfoReg", "resnet18", "resnet18_AL", "resnet18_SCPL",  "resnet18_DeInfoReg")|
|"--dataset"|"cifar10"| Dataset name </br> ("cifar10", "cifar100" or "tinyImageNet")|
|"--aug_type"|"basic"| Type of Data augmentation. Use **SCPL** augmentation like contrastive learning used. </br>("basic", "strong")|
|"--optimal"|"LARS"| Optimal Name </br> ("LARS", "SGD", "ADAM")'|
|"--epochs"|200| Number of training epochs |
|"--train_bsz"|128| Batch size of training data|
|"--test_bsz"|1024| Batch size of test data|
|"--base_lr"|0.2| Initial learning rate|
|"--end_lr"|0.002| Learning rate at the end of training|
|"--localloss"|"DeInfoReg"| Defined local loss in each layer </br> ("DeInfoReg", "contrastive")|
|"--gpus"|"0"| ID of the GPU device. If you want to use multiple GPUs, you can separate their IDs with commas, e.g., `0,1`. For single GPU models, only the first GPU ID will be used.|
|"--mlp"|"2048-2048-2048"| Size and number of layers of the Projector head |
|"--showmodelsize"|"False"| Whether show model size </br> (True, False)|
|"--train_time"|1| Round Times of training step|

---
### NLP

#### Example
```bash
# Command Example
python3 train_nlp.py --model LSTM_DeInfoReg --dataset IMDB --epochs 50 
```
#### Usage
```bash
python3 train_nlp.py [Options]
```
#### Options
| Name | Default | Description |
| -------- | -------- | -------- |
|"--task"|"nlp"|Task|
|"--model"|"LSTM_DeInfoReg"|Model Name </br> ("LSTM", "LSTM_DeInfoReg", "LSTM_SCPL", "LSTM_AL", "LSTM_DeInfoReg_side", "LSTM_DeInfoReg_Adaptive", "Transformer", "Transformer_SCPL", "Transformer_AL", "Transformer_DeInfoReg", "Transformer_DeInfoReg_side")|
|"--dataset"|"IMDB"| Dataset name </br> ("IMDB", "agnews" or "DBpedia")|
|--optimal"|"ADAM"| Optimal Name </br> ("LARS", "SGD", "ADAM")'|
|"--epochs"|50| Number of training epochs |
|"--train_bsz"|128| Batch size of training data|
|"--test_bsz"|1024| Batch size of test data|
|"--base_lr"|0.001| Initial learning rate|
|"--end_lr"|0.001| Learning rate at the end of training|
|"--localloss"|"DeInfoReg"| Defined local loss in each layer </br> ("DeInfoReg", "contrastive")|
|"--gpus"|"0"| ID of the GPU device. If you want to use multiple GPUs, you can separate their IDs with commas, e.g., `0,1`. For single GPU models, only the first GPU ID will be used.|
|"--max_len"|350| Maximum length for the sequence of input samples depend on dataset |
|"--h_dim"|300|Dimensions of the hidden layer|
|"--word_vec_type"|"pretrain"|Type of word embedding, if dont use word embedding, please set "nor" </br> (pretrain, nor)|
|"--emb_dim"|"glove"|store of word embedding|
|"--h_dim"|300|Dimension of word embedding|
|"--blockwise_total"|5|Number of layers of the model.(embedding(1) + LSTM(4)) The minimum is "2". The first layer is the pre-training embedding layer, and the latter layer is lstm or transformer.|
|"--showmodelsize"|"False"| Whether show model size </br> (True, False)|
|"--train_time"|1| Round Times of training step|
|"--patiencethreshold"|1| Threshold of inference adaptive patience count|
|"--cosinesimthreshold"|0.8| Threshold of inference adaptive cosine similarity|
|"--side_dim"|200 150| side input dimention. e.g., "200 150"|

---
### Dynamic Extended Layers with new features

#### Example
```bash
# First step (train 3 LSTM blocks)
python3 train_nlp.py --dataset IMDB --model LSTM_DeInfoReg --train_bsz 512 --epochs 50 --train_time 1 --blockwise_total 4 --side_dim 100 250
# Second step (train new layer and add new feature)
python3 train_nlp.py --dataset IMDB --model LSTM_DeInfoReg_side --train_bsz 512 --base_lr 0.0001 --end_lr 0.0001 --epochs 50 --train_time 1 --blockwise_total 4 --side_dim 100 250 --model_weights_path [The file path for the training weights(.pth) in the first step]
```
