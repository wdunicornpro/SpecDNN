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

## Details on Algorithm
### Network Community Detection Algorithms
OSLOM is a Network Community Detection algorithm proposed by Andrea Lancichinetti et al.(2011)[1]. The basic idea of OSLOM is to compare the connections in the input network with those in the corresponding random network(null model). John Palowitch et al. provided more detailed description on this approach and proposed a more sophisticated algorithm for weighted networks named CMSE[2]. While we are considering to test the criteria proposed by John Palowitch et al. for CMSE in the future, we used OSLOM for our current experiments.

### OSLOM Basics
Basically, OSLOM constructs a community by randomly selecting a node as the original community and repeatedly including nodes with 'strong connection' with the current community into the community.    
The criteria of 'strong connection' for weighted networks defined by OSLOM is: 
1. ![](https://latex.codecogs.com/gif.latex?%3Cw_%7Bij%7D%3E%3D%5Cfrac%7B2%3Cw_%7Bi%7D%3E%3Cw_%7Bj%7D%3E%7D%7B%3Cw%7Bi%7D%3E&plus;%3Cw%7Bj%7D%3E%7D)
2. ![](https://latex.codecogs.com/gif.latex?r_%7Bj%7D%28c%29%3Dp%28w_%7Bcj%7D%3Ex%29%3Dexp%28-x/%3Cw_%7Bcj%7D%3E%29)
3. ![](https://latex.codecogs.com/gif.latex?%5COmega_%7Bq%7D%28r%29%3Dp%28r_%7Bq%7D%3Cx%29%3D%5Csum_%7Bi%3Dq%7D%5E%7BN-n_%7Bc%7D%7D%5Cbinom%7BN-n_%7Bc%7D%7D%7Bi%7Dx%5Ei%281-x%29%5E%7BN-n_%7Bc%7D-i%7D)

![](https://latex.codecogs.com/gif.latex?%3Cw_%7Bi%7D%3E) is the average weight of all the edges connected to node i, and ![](https://latex.codecogs.com/gif.latex?r_%7Bq%7D) is the q-st smallest value of ![](https://latex.codecogs.com/gif.latex?%5C%7Br_%7Bj%7D%5C%7D).  
Please check the original paper if you're interested in the details of OSLOM.

### Weight Representation
Since Network Community Detection algorithms require positive weights on edges, we desiged a weight representation method, where:
1. ![](https://latex.codecogs.com/gif.latex?w%27%3D%7Cw%7C) for numerical weights
2. ![](https://latex.codecogs.com/gif.latex?w%27%3Dmean%28abs%28w%29%29) for convolution kernels

### Extraction Method
Our idea is to run OSLOM on DNNs to find strongly connected sub-network.  
In order to obtain sub-network with specialized functionalities, we perform community detection starting from one of the output units. Since neural networks are fully-connected level graphs, the only reasonable approach is to run community detection in a layer-by-layer manner. Also, because of the special structure of neural networks, the criteria of strong connection can be rewritten as:
1. ![](https://latex.codecogs.com/gif.latex?r_%7Bj%7D%28C%29%3Dexp%28-%5Cfrac%7Bm%7D%7B2%7D%5Cfrac%7B%5Csum_%7Bi%5Cin%20C%7DW_%7Bij%7D%7D%7B%5Csum_%7Bi%5Cin%20C%2Cj%5Cin%20J%7DW_%7Bij%7D%7D-%5Cfrac%7B1%7D%7B2%7D%29), where ![](https://latex.codecogs.com/gif.latex?%5C%7BW_%7Bij%7D%5C%7D) is the weight matrix, C is all the units included in the last layer and J is all the units in the current layer.
2. ![](https://latex.codecogs.com/gif.latex?%5COmega_%7Bq%7D%28r%29%3Dp%28r_%7Bq%7D%3Cx%29%3D%5Csum_%7Bi%3Dq%7D%5E%7BN-n_%7Bc%7D%7D%5Cbinom%7BN-n_%7Bc%7D%7D%7Bi%7Dx%5Ei%281-x%29%5E%7BN-n_%7Bc%7D-i%7D)

After we calculate the value of ![](https://latex.codecogs.com/gif.latex?%5C%7B%5COmega_%7Bq%7D%5C%7D), we select the smallest possible value of q where![](https://latex.codecogs.com/gif.latex?%5COmega_%7Bq%7D%3Ct%2C%5COmega_%7Bq&plus;1%7D%5Cgeq%20t) and include the first q units.  
The source code for this is as following:
```python
def OSLOM_based(weights,kernel,t=0.99):

    wts = weights[kernel].abs()
    wts = wts.view(wts.shape[0],wts.shape[1],-1).mean(2)
    n,m = wts.shape
    r = exp(-m/2*wts.sum(0)/wts.sum().item()-0.5)
    Rq = sorted(list(range(m)),key=lambda x:r[x])
    Omega = [bernoulli(r[i].item(),i+1,m) for i in Rq]
    q = 1
    for i in range(m):
        if Omega[i] > t:
            q = i
            break
        elif i == m-1:
            q = m
    return Rq[:q]
```


## Reference
[1] Lancichinetti A, Radicchi F, Ramasco JJ, Fortunato S (2011) Finding Statistically Significant Communities in Networks. PLoS ONE 6(4): e18961.  
[2] J Palowitch, S Bhamidi, AB Nobel (2018) Journal of Machine Learning Research 18: 1-48.


