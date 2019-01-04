# original training -- 99.41%
python main.py --arch LeNet_5

# prune -- 9.96%   0.029129
# python main.py --arch LeNet_5 --prune --algo OSLOM --pretrained saved_models/LeNet_5.best_origin.pth.tar

# retrain -- 30.39%
# python main.py --arch LeNet_5 --retrain --pretrained saved_models/LeNet_5.prune.OSLOM.pth.tar --lr 0.01

# specialize -- 90.20%  0.007689
python main.py --arch LeNet_5 --prune --specialize 0 --algo OSLOM --pretrained saved_models/LeNet_5.best_origin.pth.tar
# 92.38%    0.022764
python main.py --arch LeNet_5 --prune --specialize 1 --algo OSLOM --pretrained saved_models/LeNet_5.best_origin.pth.tar
# 79.64%    0.025366
python main.py --arch LeNet_5 --prune --specialize 2 --algo OSLOM --pretrained saved_models/LeNet_5.best_origin.pth.tar
# 85.31%    0.026829
python main.py --arch LeNet_5 --prune --specialize 3 --algo OSLOM --pretrained saved_models/LeNet_5.best_origin.pth.tar
# 84.98%    0.012892
python main.py --arch LeNet_5 --prune --specialize 4 --algo OSLOM --pretrained saved_models/LeNet_5.best_origin.pth.tar
# 91.08%    0.021580
python main.py --arch LeNet_5 --prune --specialize 5 --algo OSLOM --pretrained saved_models/LeNet_5.best_origin.pth.tar
# 89.20%    0.040790
python main.py --arch LeNet_5 --prune --specialize 6 --algo OSLOM --pretrained saved_models/LeNet_5.best_origin.pth.tar
# 89.72%    0.022787
python main.py --arch LeNet_5 --prune --specialize 7 --algo OSLOM --pretrained saved_models/LeNet_5.best_origin.pth.tar
# 50.15%    0.025668
python main.py --arch LeNet_5 --prune --specialize 8 --algo OSLOM --pretrained saved_models/LeNet_5.best_origin.pth.tar
# 89.31%    0.045134
python main.py --arch LeNet_5 --prune --specialize 9 --algo OSLOM --pretrained saved_models/LeNet_5.best_origin.pth.tar

# retrain -- 99.21%
python main.py --arch LeNet_5 --retrain --specialize 0 --pretrained saved_models/LeNet_5.specialize.0.OSLOM.pth.tar
# 99.79%
python main.py --arch LeNet_5 --retrain --specialize 1 --pretrained saved_models/LeNet_5.specialize.1.OSLOM.pth.tar
# 99.37%
python main.py --arch LeNet_5 --retrain --specialize 2 --pretrained saved_models/LeNet_5.specialize.2.OSLOM.pth.tar
# 99.63%
python main.py --arch LeNet_5 --retrain --specialize 3 --pretrained saved_models/LeNet_5.specialize.3.OSLOM.pth.tar
# 99.34%
python main.py --arch LeNet_5 --retrain --specialize 4 --pretrained saved_models/LeNet_5.specialize.4.OSLOM.pth.tar
# 98.53%
python main.py --arch LeNet_5 --retrain --specialize 5 --pretrained saved_models/LeNet_5.specialize.5.OSLOM.pth.tar
# 99.73%
python main.py --arch LeNet_5 --retrain --specialize 6 --pretrained saved_models/LeNet_5.specialize.6.OSLOM.pth.tar
# 99.56%
python main.py --arch LeNet_5 --retrain --specialize 7 --pretrained saved_models/LeNet_5.specialize.7.OSLOM.pth.tar
# 99.56%
python main.py --arch LeNet_5 --retrain --specialize 8 --pretrained saved_models/LeNet_5.specialize.8.OSLOM.pth.tar
# 99.05%
python main.py --arch LeNet_5 --retrain --specialize 9 --pretrained saved_models/LeNet_5.specialize.9.OSLOM.pth.tar

