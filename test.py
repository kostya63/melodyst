import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import pathlib
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils import mkldnn as mkldnn_utils
from torchvision.io import read_image
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from pytorch_metric_learning import losses, miners, distances, reducers, regularizers
#from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torchaudio
from torchaudio.transforms import  MelSpectrogram, Resample, AmplitudeToDB
from prettytable import PrettyTable
import my_net as mn
import my_sampler as ms
from sklearn.manifold import TSNE
import plotly.express as px


#------------------------------------------
batch_size = 1
test_batch_size = 1
path = "model-full-w4-emb128-100.pth"
knear = 3
inner_dim = 16384
embedding_dim = 128
sound_len = 8
resample_rate = 14000
tensor_size = sound_len * resample_rate
target_tensor_size = 544

n_mels = 128
n_mfcc = 32
resample_rate = 14000
n_fft = 2048
hop_length = 626

w_factor = 4
L1 = int(8 * w_factor)
L2 = int(32 * w_factor)
L3 = int(128 * w_factor)
L4 = int(512 * w_factor)

classes = [
     "Sound_Guitar",
     "Sound_Drum",
     "Sound_Violin",
     "Sound_Piano"
]
   
correct_answers = [[0, 0], [0, 0], [0, 0], [0, 0]]

#---------------------------------------------------------------------
# Let's go!
#---------------------------------------------------------------------

# Create data loaders.
train_dataset = ms.YesDataset('/home/mkr/dl3/4instruments/Metadata_Train.csv', '/home/mkr/dl3/4instruments/Train_submission', sample_rate = resample_rate, sound_len=sound_len, n_mels = n_mels, USE_AGUMENTATION = False)
test_dataset = ms.YesDataset('/home/mkr/dl3/4instruments/Metadata_Test.csv', '/home/mkr/dl3/4instruments/Test_submission', sample_rate = resample_rate, sound_len=sound_len, n_mels = n_mels, USE_AGUMENTATION = False)

m=len(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#init the model
model = mn.wrn_nn(L1=L1, L2=L2, L3=L3, L4=L4, L5=0, inner_dim=inner_dim, embedding_dim=embedding_dim).to(device)

checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])

encoded_samples = []
for batch, (a, b) in enumerate(test_loader):
    a = a.to(device)
    label = b
    # Encode image
    model.eval()
    with torch.no_grad():
        encoded_img = model(a)
    # Append to list
    encoded_img = encoded_img.flatten().cpu().numpy()
    encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
    encoded_sample['label'] = label
    encoded_samples.append(encoded_sample)

encoded_samples = pd.DataFrame(encoded_samples)

tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
fig = px.scatter(tsne_results, x=0, y=1, color=encoded_samples.label.astype(str), labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
fig.show()