import numpy as np
import matplotlib.pyplot as plt

import utils.model as model
import pickle

import glob
import os

import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

os.makedirs("results", exist_ok=True)
use_cuda=True
init_num_filters=64
inter_fc_dim=384
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
            cnn = VGG()  # Use your VGG model for clean models
        else:
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

