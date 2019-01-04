# original training -- 98.28%
python main.py

# prune -- 16.35%  0.009316
# python main.py --prune --algo OSLOM --pretrained saved_models/LeNet_300_100.best_origin.pth.tar

# retrain -- 20.13%
# python main.py --retrain --pretrained saved_models/LeNet_300_100.prune.OSLOM.pth.tar --lr 0.01

# specialize -- 90.21%  0.005015
python main.py --prune --specialize 0 --algo OSLOM --pretrained saved_models/LeNet_300_100.best_origin.pth.tar
# 79.04%    0.000770
python main.py --prune --specialize 1 --algo OSLOM --pretrained saved_models/LeNet_300_100.best_origin.pth.tar
# 89.68%    0.002014
python main.py --prune --specialize 2 --algo OSLOM --pretrained saved_models/LeNet_300_100.best_origin.pth.tar
# 64.92%    0.001262
python main.py --prune --specialize 3 --algo OSLOM --pretrained saved_models/LeNet_300_100.best_origin.pth.tar
# 88.07%    0.003388
python main.py --prune --specialize 4 --algo OSLOM --pretrained saved_models/LeNet_300_100.best_origin.pth.tar
# 91.29%    0.002618
python main.py --prune --specialize 5 --algo OSLOM --pretrained saved_models/LeNet_300_100.best_origin.pth.tar
# 62.39%    0.001349
python main.py --prune --specialize 6 --algo OSLOM --pretrained saved_models/LeNet_300_100.best_origin.pth.tar
# 77.60%    0.004343
python main.py --prune --specialize 7 --algo OSLOM --pretrained saved_models/LeNet_300_100.best_origin.pth.tar
# FAIL(53.14%   0.001315)
python main.py --prune --specialize 8 --algo OSLOM --pretrained saved_models/LeNet_300_100.best_origin.pth.tar
# FAIL(85.49%   0.004467)
python main.py --prune --specialize 9 --algo OSLOM --pretrained saved_models/LeNet_300_100.best_origin.pth.tar

# retrain -- 98.90%
python main.py --retrain --specialize 0 --pretrained saved_models/LeNet_300_100.specialize.0.OSLOM.pth.tar
# 98.66%
python main.py --retrain --specialize 1 --pretrained saved_models/LeNet_300_100.specialize.1.OSLOM.pth.tar
# 98.47%
python main.py --retrain --specialize 2 --pretrained saved_models/LeNet_300_100.specialize.2.OSLOM.pth.tar
# 97.23%
python main.py --retrain --specialize 3 --pretrained saved_models/LeNet_300_100.specialize.3.OSLOM.pth.tar
# 99.01%
python main.py --retrain --specialize 4 --pretrained saved_models/LeNet_300_100.specialize.4.OSLOM.pth.tar
# 98.46%
python main.py --retrain --specialize 5 --pretrained saved_models/LeNet_300_100.specialize.5.OSLOM.pth.tar
# 97.29%
python main.py --retrain --specialize 6 --pretrained saved_models/LeNet_300_100.specialize.6.OSLOM.pth.tar
# 98.18%
python main.py --retrain --specialize 7 --pretrained saved_models/LeNet_300_100.specialize.7.OSLOM.pth.tar
# 96.88%
python main.py --retrain --specialize 8 --pretrained saved_models/LeNet_300_100.specialize.8.OSLOM.pth.tar
# 98.70%
python main.py --retrain --specialize 9 --pretrained saved_models/LeNet_300_100.specialize.9.OSLOM.pth.tar
