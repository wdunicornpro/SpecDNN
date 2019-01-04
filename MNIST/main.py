from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'/../')
import models
from torchvision import datasets, transforms
from torch.autograd import Variable
from util import *
import prune,specialize


def save_state(model, acc):
    print('==> Saving model ...')
    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')
    state = {
            'acc': acc,
            'state_dict': model.state_dict(),
            'mask':model.mask
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    if args.prune:
        if args.specialize != None:
            torch.save(state, 'saved_models/%s.specialize.%d.%s.pth.tar'%(args.arch,args.specialize,args.algo))
        else:
            torch.save(state, 'saved_models/'+args.arch+'.prune.'+\
                args.algo+'.pth.tar')
    elif args.retrain:
        if args.specialize != None:
            torch.save(state, 'saved_models/%s.specialize.%d.best_retrained.pth.tar'%(args.arch,args.specialize))
        else:
            torch.save(state, 'saved_models/'+args.arch+'.best_retrained.pth.tar')
    else:
        torch.save(state, 'saved_models/'+args.arch+'.best_origin.pth.tar')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.specialize != None:
            target = (target == args.specialize).type(torch.Tensor)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        if args.specialize != None:
            output = output[:,args.specialize]
        # print(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return

def test(evaluate=False):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if args.specialize != None:
            target = (target == args.specialize).type(torch.Tensor)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        with torch.no_grad():
            output = model(data)
            if args.specialize != None:
                output = output[:,args.specialize]
            # print(output)
            # print(target)
            test_loss += criterion(output, target).item()
            if args.specialize != None:
                pred = output.data > 0.5
                correct += pred.eq(target.data.byte()).sum()
            else:
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # correct += sum([output[i][target[i]]>0.5 for i in range(len(target))])
    
    acc = 100. * float(correct) / len(test_loader.dataset)
    if (args.prune and not args.retrain) or (acc > best_acc):
        best_acc = acc
        if not evaluate:
            save_state(model, best_acc)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        acc))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return

if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
            help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
            help='number of epochs to train (default: 50)')
    parser.add_argument('--lr-epochs', type=int, default=15, metavar='N',
            help='number of epochs to decay the lr (default: 15)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
            help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
            metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='LeNet_300_100',
            help='the MNIST network structure: LeNet_300_100 | LeNet_5')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=False,
            help='whether to run evaluation')
    parser.add_argument('--retrain', action='store_true', default=False,
            help='retrain the pruned network')
    parser.add_argument('--prune', action='store_true', default=False,
            help='whether to prune the network')
    parser.add_argument('--specialize', type=int, metavar='N', default=None,
            help='extract the specialized network')
    parser.add_argument('--algo',default=None,
            help='clustering algorithm to use')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print_args(args)
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # generate the model
    if args.arch == 'LeNet_300_100':
        model = models.LeNet_300_100()
    elif args.arch == 'LeNet_5':
        model = models.LeNet_5()
    else:
        print('ERROR: specified arch is not suppported')
        exit()

    if not args.pretrained:
        best_acc = 0.0
    else:
        pretrained_model = torch.load(args.pretrained)
        # best_acc = pretrained_model['acc'] if not args.algo else 0
        best_acc = 0.0
        mask = pretrained_model['mask'] if 'mask' in pretrained_model else []
        model.mask = mask
        load_state(model, pretrained_model['state_dict'])

    if args.cuda:
        model.cuda()

    print(model)
    param_dict = dict(model.named_parameters())
    params = []
    
    for key, value in param_dict.items():
        if 'mask' in key:
            params += [{'params':[value], 'lr': args.lr,
                'momentum':args.momentum,
                'weight_decay': 0.0,
                'key':key}]
        else:
            params += [{'params':[value], 'lr': args.lr,
                'momentum':args.momentum,
                'weight_decay': args.weight_decay,
                'key':key}]

    # load data
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)
    if args.specialize != None:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer,args.lr_epochs, gamma=0.1)
    if args.evaluate:
        print_layer_info(model)
        test(evaluate=True)
        exit()
    
    if args.prune:
        if args.specialize != None:
            print("Specializing using %s..."%args.algo)
            print_layer_info(model)
            if args.arch == 'LeNet_300_100':
                specialize.specialize_lenet_300_100(args.algo,model,args.specialize)
            elif args.arch == 'LeNet_5':
                specialize.specialize_lenet_5(args.algo,model,args.specialize)
        else:
            print("Pruning using %s..."%args.algo)
            print_layer_info(model)
            if args.arch == 'LeNet_300_100':
                prune.prune_lenet_300_100(args.algo,model)
            elif args.arch == 'LeNet_5':
                prune.prune_lenet_5(args.algo,model)
        test()
        exit()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
       

    # train
    for epoch in range(1, args.epochs + 1):
        print('Learning Rate:',optimizer.param_groups[0]['lr'])
        train(epoch)
        scheduler.step()
        test()
