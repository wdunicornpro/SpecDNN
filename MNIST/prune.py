import argparse
import os,sys
cwd = os.getcwd()
sys.path.append(cwd+'/../')
import NeuronClustering
import torch


def prune_lenet_300_100(algo,model):

    if algo == 'OSLOM':
        algorithm = NeuronClustering.OSLOM_based
    elif algo == 'CCME':
        algorithm = NeuronClustering.CCME_based
    print("pruning ip3...")
    n3 = algorithm(model.ip3.weight,list(range(model.ip3.in_features)))
    print("pruning ip2...")
    n2 = algorithm(model.ip2.weight,n3)
    print("pruning ip1...")
    n1 = algorithm(model.ip1.weight,n2)
    print("Compress Rate: %f"%((10*len(n3)+len(n3)*len(n2)+len(n2)*len(n1))/(10*model.ip3.in_features+model.ip3.in_features*model.ip2.in_features+model.ip2.in_features*model.ip1.in_features)))
    model.mask = [
        list(set(range(model.ip1.in_features)) - set(n1)),
        list(set(range(model.ip2.in_features)) - set(n2)),
        list(set(range(model.ip3.in_features)) - set(n3))
    ]
    print("Feature Pooling Rate: %f"%(len(n1)/model.ip1.in_features))

def prune_lenet_5(algo,model):
    
    if algo == 'OSLOM':
        algorithm = NeuronClustering.OSLOM_based
    elif algo == 'CCME':
        algorithm = NeuronClustering.CCME_based
    
    print('pruning ip2...')
    n4 = algorithm(model.ip2.weight,list(range(model.ip2.in_features)))
    print('pruning ip1...')
    n3 = algorithm(model.ip1.weight,n4)
    print('pruning con2...')
    n2 = algorithm(model.conv2.weight,n3)    
    print('pruning con1...')
    n1 = algorithm(model.conv1.weight,n2)
    # print(n1)
    model.mask = [
        list(set(range(model.conv1.in_channels)) - set(n1)),
        list(set(range(model.conv2.in_channels)) - set(n2)),
        list(set(range(model.ip1.in_features)) - set(n3)),
        list(set(range(model.ip2.in_features)) - set(n4))
    ]

    print("Compress Rate: %f"%(
        (
            model.conv1.weight.view(-1).shape[0]*len(n1)/model.conv1.in_channels
            +model.conv2.weight.view(-1).shape[0]*len(n2)/model.conv2.in_channels
            +model.ip1.weight.view(-1).shape[0]*len(n3)/model.ip1.in_features
            +model.ip2.weight.view(-1).shape[0]*len(n4)/model.ip2.in_features)
            /(
            model.conv1.weight.view(-1).shape[0]
            +model.conv2.weight.view(-1).shape[0]
            +model.ip1.weight.view(-1).shape[0]
            +model.ip2.weight.view(-1).shape[0]))
        )

    print("Feature Pooling Rate: %f"%(len(n2)/model.conv2.in_channels))

def mask(masked,x):
    with torch.no_grad():
        x[:,masked] = torch.zeros_like(x[:,masked])
    return x