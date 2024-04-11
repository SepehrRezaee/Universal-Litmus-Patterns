import numpy as np
import matplotlib.pyplot as plt

import utils.model as model
import pickle

import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, init_num_filters=64, inter_fc_dim=384, nofclasses=10, nofchannels=3):
        super(VGG, self).__init__()
        self.init_num_filters_ = init_num_filters
        self.inter_fc_dim_ = inter_fc_dim
        self.nofclasses_ = nofclasses

        self.features = nn.Sequential(
            nn.Conv2d(nofchannels, init_num_filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(init_num_filters),
            nn.ReLU(True),

            nn.Conv2d(init_num_filters, init_num_filters, kernel_size=5, padding=2),
            nn.BatchNorm2d(init_num_filters),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(init_num_filters, init_num_filters*2, kernel_size=5, padding=2),
            nn.BatchNorm2d(init_num_filters*2),
            nn.ReLU(True),

            nn.Conv2d(init_num_filters*2, init_num_filters*2, kernel_size=5, padding=2),
            nn.BatchNorm2d(init_num_filters*2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Add or adjust layers to match the architecture of the saved model
        )

        self.fc = nn.Sequential(
            nn.Linear(init_num_filters * 2 * 8 * 8, inter_fc_dim),  # Adjust the input size accordingly
            nn.BatchNorm1d(inter_fc_dim),
            nn.ReLU(True),
            nn.Dropout(p=0.2),

            nn.Linear(inter_fc_dim, inter_fc_dim // 2),
            nn.BatchNorm1d(inter_fc_dim // 2),
            nn.ReLU(True),
            nn.Dropout(p=0.2),

            nn.Linear(inter_fc_dim // 2, nofclasses),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers
        x = self.fc(x)
        return x



def create_vgg():
    return VGG()


def test():
    net = create_vgg()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())


if __name__ == "__main__":
    test()



os.makedirs("results", exist_ok=True)
use_cuda=True
init_num_filters=32
inter_fc_dim=128
nofclasses=10 #CIFAR10


# poisoned
poisoned_models_test = sorted(glob.glob('/kaggle/working/poisoned_vggmod_CIFAR-10_0100.pth.tar'))

# clean models
clean_models=glob.glob('/kaggle/input/wanetattack-cifar10/Wanet_dataset/CIFAR10/clean/*.pth.tar')

# val - 100 clean 100 poisoned
models_test=clean_models + poisoned_models_test
labels_test=np.concatenate([np.zeros((len(clean_models),)),np.ones((len(poisoned_models_test),))])

cnn=model.CNN_classifier(init_num_filters=init_num_filters,
                         inter_fc_dim=inter_fc_dim,nofclasses=nofclasses,
                         nofchannels=3,use_stn=False)
if use_cuda:
    device=torch.device('cuda')
    cnn.cuda()
else:
    device=torch.device('cpu')


def getLogit(cnn,ulps,W,b,device):
    logit=torch.matmul(cnn(ulps.to(device)).view(1,-1),W)+b
    return logit


# Later, when you get to the point of computing logits
def getLogitCu(cnn, ulps, W, b, device):
    # Ensure inputs are also moved to the correct device
    ulps = ulps.to(device)
    W = W.to(device)
    b = b.to(device)

    # print(ulps.shape)

    # Now that everything is on the same device, perform your operations
    logit = torch.matmul(cnn(ulps).view(1, -1), W) + b
    return logit

# Example usage
# Ensure `ulps`, `W`, and `b` are tensors that can be moved to the device
# logit = getLogit(cnn, ulps, W, b, device)



# # load baseline mat data
# file = "ROC_CIFAR.mat"
# from scipy.io import loadmat
# x = loadmat(file)
#
#
# y_true = x['y'].squeeze()
# y_score = x['s'].squeeze()

plt.figure(figsize=(14,10))
plt.plot([0, 1], [0, 1], linestyle='--',linewidth=3)

auc = list()
# for N in [1, 5, 10]:
# for N in [10]:
#     ulps, W, b = pickle.load(open('/kaggle/working/CIFAR10_best_universal_image_diff_dist_N{}.pkl'.format(N),'rb'))
#     features=list()
#     probabilities=list()
#     for i,model_ in enumerate(models_test):
#         cnn.load_state_dict(torch.load(model_))
#         cnn.eval()
#         label=np.array([labels_test[i]])
#         logit=getLogit(cnn,ulps,W,b,device)
#         probs=torch.nn.Softmax(dim=1)(logit)
#         features.append(logit.detach().cpu().numpy())
#         probabilities.append(probs.detach().cpu().numpy())


#     features_np=np.stack(features).squeeze()
#     probs_np=np.stack(probabilities).squeeze()


#     fpr, tpr, thresholds=roc_curve(labels_test,probs_np[:,1])
#     auc = roc_auc_score(labels_test,probs_np[:,1])

#     pickle.dump([fpr, tpr, thresholds, auc], open("./results/ROC_ULP_N{}.pkl".format(N), "wb"))


# Evaluate Universal Litmus Patterns (ULP)
for N in [10]:
    ulps, W, b = pickle.load(open(f'/kaggle/working/CIFAR10_best_universal_image_diff_dist_N{N}.pkl', 'rb'))
    features = []
    probabilities = []

    for i, model_path in enumerate(models_test):
        # Initialize the appropriate model based on the type of model being evaluated
        if model_path in clean_models:
            print(model_path)
            cnn1 = create_vgg()
            cnn1.to(device)

            cnn1.load_state_dict(torch.load(model_path, map_location=device)['netC'], strict=False)
            cnn1.eval()

            logit = getLogitCu(cnn1, ulps, W, b, device)
            probs = torch.nn.Softmax(dim=1)(logit)
            features.append(logit.detach().cpu().numpy())
            probabilities.append(probs.detach().cpu().numpy())

        else:
            print(model_path)
            cnn = model.CNN_classifier()  # Use the existing model for poisoned models
        
            cnn.to(device)
            cnn.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'], strict=False)
            cnn.eval()

            logit = getLogit(cnn, ulps, W, b, device)
            probs = torch.nn.Softmax(dim=1)(logit)
            features.append(logit.detach().cpu().numpy())
            probabilities.append(probs.detach().cpu().numpy())

    features_np = np.stack(features).squeeze()
    probs_np = np.stack(probabilities).squeeze()

    fpr, tpr, thresholds = roc_curve(labels_test, probs_np[:, 1])
    auc = roc_auc_score(labels_test, probs_np[:, 1])

    pickle.dump([fpr, tpr, thresholds, auc], open(f"./results/ROC_ULP_N{N}.pkl", "wb"))




