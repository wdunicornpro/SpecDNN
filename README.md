# SpecDNN
A Pytorch implementation of Specialized DNN Extraction based on Social Network Community Detection.

## Introduction
The purpose of this project is to extract light-weighted specialized sub-network with specific functionalities from a heavy-weighted general-purposed pre-trained neural network. One practical application of this project will be extracting application specific neural network models to run and test on local mobile devices.  
The basic idea of this project is to adapt Social Network Community Detection methods to neural networks. The community detection algorithm we choose is OSLOM(https://oslom.org).  
We tested our method on MNIST using LeNet-5 and LeNet-300-100.


## Usage
To run the whole experiment, run:
```sh
$ cd MNIST
$ ./spec.lenet_5.sh
$ ./spec.lenet_300_100.sh
```
The script will train the model, run Spec DNN Extraction for the 10 classes, fine tune and test the resulting 10 specialized models.
```sh
# original training -- 98.28%
python main.py
# specialize -- 90.21%  0.005015
python main.py --prune --specialize 0 --algo OSLOM --pretrained saved_models/LeNet_300_100.best_origin.pth.tar
# retrain -- 98.90%
python main.py --retrain --specialize 0 --pretrained saved_models/LeNet_300_100.specialize.0.OSLOM.pth.tar
```

## Results
Here is the results for the experiment:
### LeNet-300-100 on MNIST
||0-spec|1-spec|2-spec|3-spec|4-spec|5-spec|6-spec|7-spec|8-spec|9-spec|spec avg|original|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Compress Rate |0.5015%|0.0770%|0.2014%|0.1262%|0.3388%|0.2618%|0.1349%|0.4343%|0.1315%|0.4467%|0.26541%|100%|
|Accuracy|98.9%|98.66%|98.47%|97.23%|99.01%|98.46%|97.29%|98.18%|96.88%|98.70%|98.12%|98.28%|

### LeNet-5 on MNIST
||0-spec|1-spec|2-spec|3-spec|4-spec|5-spec|6-spec|7-spec|8-spec|9-spec|spec avg|original|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Compress Rate |0.7689%|2.2764%|2.5366%|2.6829%|1.2892%|2.1580%|4.0790%|2.2787%|2.5668%|4.5134%|2.515%|100%|
|Accuracy|99.21%|99.79%|99.37%|99.63%|99.34%|98.53%|99.73%|99.56%|99.56%|99.05%|99.377|99.41%|
