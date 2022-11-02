import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import pathlib
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from pytorch_metric_learning import losses, miners, distances, reducers, regularizers
#from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torchaudio
from torchaudio.transforms import  MelSpectrogram, Resample, AmplitudeToDB
from nnAudio import Spectrogram
from prettytable import PrettyTable

#------------------------------------------
batch_size = 28
#1 - 10 learning_rate = 0.00001
#11 - 20 learning_rate = 0.000009
#21 - 30 learning_rate = 0.000008
#31 - 40 learning_rate = 0.000007
#41 - 50 learning_rate = 0.000006
#51 - 60 learning_rate = 0.000005
#61 - 70 learning_rate = 0.000004
#71 - 80 learning_rate = 0.000003
#81 - 90 learning_rate = 0.000002
#91 - 100 learning_rate = 0.000001
#starting rate 1e-5
#learning_rate = 1e-5
#for linear learning scheduling
learning_rate = 0.000001 
#learning_rate = 0.00001  * pow(0.5, 20)
scheduler_step_size = 15
scheduler_gamma = 0.5
use_scheduling = True

#for triangle learning scheduling
min_lr =  0.000001
max_lr = 0.00001
step_size_up = 135
w_decay = 1e-7
test_batch_size = 1
epochs = 10
knear = 5

inner_dim = 100
embedding_dim = 2
sound_len = 20
resample_rate = 14000
tensor_size = sound_len * resample_rate
target_tensor_size = 544


class RandomSpeedChange:
	def __init__(self, sample_rate):
		self.sample_rate = sample_rate
	
	def __call__(self, audio_data):
		speed_factor = random.choice(["0,9", "0,95", "1,05", "1,1"])
		# change speed and resample to original rate
		sox_effects = [["speed", speed_factor], ["rate", str(self.sample_rate)],]
		transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(tensor = audio_data, sample_rate = self.sample_rate, effects = sox_effects)
		return transformed_audio

class RandomBackgroundNoise:
    def __init__(self, sample_rate, noise_dir, min_snr_db=10, max_snr_db=15):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if not os.path.exists(noise_dir):
            raise IOError(f'Noise directory `{noise_dir}` does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))
        if len(self.noise_files_list) == 0:
            raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

    def __call__(self, audio_data):
        random_noise_file = random.choice(self.noise_files_list)
        effects = [
            ['remix', '1'], # convert to mono
            ['rate', str(self.sample_rate)], # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length - audio_length)
            noise = noise[..., offset:offset+audio_length]
        elif noise_length < audio_length:
            noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length - noise_length))], dim = -1)

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power

        return (scale * audio_data + noise ) / 2

class YesDataset(Dataset):
	def __init__(self, annotations_file, sound_dir, transform=None, target_transform=None):
		self.sound_labels = pd.read_csv(annotations_file)
		self.sound_dir = sound_dir
		self.transform = transform
		self.target_transform = target_transform
	def __len__(self):
		return len(self.sound_labels)
	
	def __getitem__(self, idx):
		sound_path = os.path.join(self.sound_dir, self.sound_labels.iloc[idx, 0])
		torchaudio.set_audio_backend("sox_io")
		sound, original_sample_rate = torchaudio.load(sound_path)
		#sample_rate = sound_t[1]
		#resampler = Resample(sample_rate, resample_rate)
		#sound = sound_t[0]
		#sound = torch.mean(sound, 0)
		
		#convert to mono and resample
		effects = [['remix', '1'], ['rate', str(resample_rate)],]
		sound, _ = torchaudio.sox_effects.apply_effects_tensor(sound, resample_rate, effects)
		
		# source sound agumentation
		agumentation = random.choice(['nothing', 'speed', 'noise', 'speed + noise'])
		#agumentation = random.choice(['none', 'speed'])
		#agumentation = random.choice(['nothing', 'noise'])
		#agumentation = 'none'
		if agumentation == 'speed':
			speed_transform = RandomSpeedChange(resample_rate)
			sound = speed_transform(sound)
		elif agumentation == 'noise':
			noise_transform = RandomBackgroundNoise(resample_rate, './data/noise')
			sound = noise_transform(sound)
		elif agumentation == 'speed + noise':
			speed_transform = RandomSpeedChange(resample_rate)
			sound = speed_transform(sound)
			noise_transform = RandomBackgroundNoise(resample_rate, './data/noise')
			sound = noise_transform(sound)
			
		# apply 20 sec lenght
		sound = sound.squeeze(0)
		if sound.size()[0]  <  tensor_size:
			sound = nn.ConstantPad1d((0, tensor_size - sound.size()[0] ), 0)(sound)
		elif sound.size()[0]  >  tensor_size:
			sound = sound.narrow(0, 0, tensor_size)
		
		# Perform transformation
		label = self.sound_labels.iloc[idx, 1]
		if self.transform:
			sound = self.transform(sound)
		if self.target_transform:
			label = self.target_transform(label)
		
		#convert to spectrogram
		sound = sound.unsqueeze(0)
		#mspectre1 = MelSpectrogram(sample_rate=resample_rate, n_fft=n_fft1, hop_length=hop_length1, center=True, pad_mode="reflect", power=2.0, norm='slaney', onesided=True, n_mels=n_mels, mel_scale="htk")
		decibel = AmplitudeToDB(top_db=80)
		if USE_TRIPLE_SPECS:
			mspectre1 = MelSpectrogram(sample_rate=resample_rate, n_fft=n_fft1, hop_length=hop_length1, center=True, pad_mode="reflect", power=2.0, onesided=True, n_mels=n_mels, mel_scale="htk")
			mspectre2 = MelSpectrogram(sample_rate=resample_rate, n_fft=n_fft2, hop_length=hop_length2, center=True, pad_mode="reflect", power=2.0, onesided=True, n_mels=n_mels, mel_scale="htk")
			mspectre3 = MelSpectrogram(sample_rate=resample_rate, n_fft=n_fft3, hop_length=hop_length3, center=True, pad_mode="reflect", power=2.0, onesided=True, n_mels=n_mels, mel_scale="htk")
			sound1 = mspectre1(sound)
			sound1 = decibel(sound1)
			sound2 = mspectre2(sound)
			sound2 = decibel(sound2)
			sound2 = nn.ConstantPad1d((0, sound1.size()[1] - sound2.size()[1] ), 0)(sound2)
			sound3 = mspectre3(sound)
			sound3 = decibel(sound3)
			sound3 = nn.ConstantPad1d((0, sound1.size()[1] - sound3.size()[1] ), 0)(sound3)
			sound = torch.stack((sound1, sound2, sound3))
		else:
			mspectre = MelSpectrogram(sample_rate=resample_rate, n_fft=n_fft1, hop_length=hop_length1, center=True, pad_mode="reflect", power=2.0, onesided=True, n_mels=n_mels, mel_scale="htk")
			sound = mspectre(sound)
			sound = decibel(sound)
			#img = sound.squeeze(0)
			#plt.imshow(img)
		return sound, label
     
class audio_nn(nn.Module):
   def __init__(self):
	   super(audio_nn, self).__init__()
	   #self.mel = Spectrogram.MelSpectrogram(sr=resample_rate, n_fft=1024, hop_length=512, n_mels=128, trainable_mel=False, trainable_STFT=False)
	   self.BNR_L1 = nn.Sequential(nn.BatchNorm2d(L1), nn.ReLU())
	   self.conv_k1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=L1, kernel_size=(25, 5), stride=(2, 1), padding=(4, 2), bias=False), nn.BatchNorm2d(L1))
	   self.conv_k2 = nn.Sequential(nn.Conv2d(in_channels=L1, out_channels=L1, kernel_size=(15, 3), stride=(2, 1), padding=(3, 1), bias=False), nn.BatchNorm2d(L1))
	   self.conv_k3 = nn.Sequential(nn.Conv2d(in_channels=L1, out_channels=L1, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1), bias=False), nn.BatchNorm2d(L1))
	   self.conv_k4 = nn.Sequential(nn.Conv2d(in_channels=L1, out_channels=L2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(L2))
	   self.conv_k5 = nn.Sequential(nn.Conv2d(in_channels=L2, out_channels=L3, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False), nn.BatchNorm2d(L3))
	   self.conv_k6 = nn.Sequential(nn.Conv2d(in_channels=L3, out_channels=L3, kernel_size=(2, 2), stride=(2, 1), padding=(0, 0), bias=False), nn.BatchNorm2d(L3))
	   self.adaptive_pool2d = nn.AdaptiveAvgPool2d((1, 1))
	   self.lin1 = nn.Sequential(nn.Flatten(), nn.Linear(inner_dim, embedding_dim))
	   #self.apply(weights_init_kaiming)
	   #self.apply(fc_init_weights)
        
   def init_weights(self):
		   for module in self.modules():
			   if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
				   nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')

   def forward(self, x):
	   #act = F.tanh
	   act = nn.LeakyReLU(0.1)
	   #print('input', x.size())
	   x = self.conv_k1(x)
	   x = act(x)
	   	   
	   #print('L1', x.size())
	   #L1
	   x = self.conv_k2(x)
	   x = act(x)
	   
	   #print('L2', x.size())
	   #L2
	   x = self.conv_k2(x)
	   x = act(x)
	   
	   #print('L3', x.size())
	   #L3
	   x = self.conv_k3(x)
	   x = act(x)
	   
	   #print('L4', x.size())
	   #L4
	   x = self.conv_k4(x)
	   x = act(x)
	   	   
	   #L5
	   #print('L5', x.size())
	   x = self.conv_k5(x)
	   x = act(x)
	   
	   #L6
	   #print('L6', x.size())
	   x = self.conv_k6(x)
	   x = act(x)
	   
	   #print('before pooling', x.size())
	   x = self.adaptive_pool2d(x)
	   #print('before flatten', x.size())
	   x = self.lin1(x)
	   return x


class wrn_nn(nn.Module):
   def __init__(self, L1, L2, L3, L4, L5, inner_dim, embedding_dim):
	   super(wrn_nn, self).__init__()
	   self.L1 = L1
	   self.L2 = L2
	   self.L3 = L3
	   self.L4 = L4
	   self.L5 = L5
	   self.inner_dim = inner_dim
	   self.embedding_dim = embedding_dim
	   #self.mel = Spectrogram.MelSpectrogram(sr=resample_rate, n_fft=1024, hop_length=512, n_mels=128, trainable_mel=False, trainable_STFT=False)
	   self.BNR_L1 = nn.Sequential(nn.BatchNorm2d(L1), nn.ReLU())
	   self.conv_1_L1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=L1, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(self.L1))
	   self.conv_L1_L1_3_keep = nn.Sequential(nn.Conv2d(in_channels=self.L1, out_channels=self.L1, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L1))
	   self.conv_L1_L2_1_down = nn.Sequential(nn.Conv2d(in_channels=self.L1, out_channels=self.L2, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L2))
	   self.conv_L1_L2_3_down = nn.Sequential(nn.Conv2d(in_channels=self.L1, out_channels=self.L2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.L2))
	   self.conv_L2_L2_3_keep = nn.Sequential(nn.Conv2d(in_channels=self.L2, out_channels=self.L2, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L2))
	   self.conv_L2_L3_1_down = nn.Sequential(nn.Conv2d(in_channels=self.L2, out_channels=self.L3, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L3))
	   self.conv_L2_L3_3_down = nn.Sequential(nn.Conv2d(in_channels=self.L2, out_channels=self.L3, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.L3))
	   self.conv_L3_L3_3_keep = nn.Sequential(nn.Conv2d(in_channels=self.L3, out_channels=self.L3, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L3))
	   self.conv_L3_L4_1_down = nn.Sequential(nn.Conv2d(in_channels=self.L3, out_channels=self.L4, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L4))
	   self.conv_L3_L4_3_down = nn.Sequential(nn.Conv2d(in_channels=self.L3, out_channels=self.L4, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.L4))
	   self.conv_L4_L4_3_keep = nn.Sequential(nn.Conv2d(in_channels=self.L4, out_channels=self.L4, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L4))
	   self.conv_L4_L5_1_down = nn.Sequential(nn.Conv2d(in_channels=self.L4, out_channels=self.L5, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L5))
	   self.conv_L4_L5_3_down = nn.Sequential(nn.Conv2d(in_channels=self.L4, out_channels=self.L5, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.L5))
	   self.conv_L5_L5_3_keep = nn.Sequential(nn.Conv2d(in_channels=self.L5, out_channels=self.L5, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L5))
	   self.adaptive_pool2d = nn.AdaptiveAvgPool2d((1, 1))
	   self.lin1 = nn.Sequential(nn.Flatten(), nn.Linear(self.inner_dim, self.embedding_dim))
	   #self.apply(weights_init_kaiming)
	   #self.apply(fc_init_weights)
        
   def init_weights(self):
		   for module in self.modules():
			   if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
				   nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')

   def forward(self, x):
	   #act = F.tanh
	   act = nn.LeakyReLU(0.1)
	   #print('input', x.size())
	   x = self.conv_1_L1(x)
	   x = act(x)
	   x = F.max_pool2d(x, 2)
	   res = x
	   
	   #print('L1', x.size())
	   #L1/1
	   x = self.conv_L1_L1_3_keep(x)
	   x = act(x)
	   x = self.conv_L1_L1_3_keep(x) + res
	   x = act(x)
	   res = x
	   
	   #L1/2
	   x = self.conv_L1_L1_3_keep(x)
	   x = act(x)
	   x = self.conv_L1_L1_3_keep(x) + res
	   x = act(x)
	   res = x
	   
	   #L1/3
	   x = self.conv_L1_L1_3_keep(x)
	   x = act(x)
	   x = self.conv_L1_L1_3_keep(x) + res
	   x = act(x)
	   res = x

	   #L1/4
	   x = self.conv_L1_L1_3_keep(x)
	   x = act(x)
	   x = self.conv_L1_L1_3_keep(x) + res
	   x = act(x)	   
	   	   
	   res = self.conv_L1_L2_1_down(x)
	   
	   #print('L2', res.size())
	   #L2/1
	   x = self.conv_L1_L2_3_down(x)
	   x = act(x)
	   x = self.conv_L2_L2_3_keep(x) + res
	   x = act(x)
	   res = x
	   
	   #L2/2
	   x = self.conv_L2_L2_3_keep(x)
	   x = act(x)
	   x = self.conv_L2_L2_3_keep(x) + res
	   x = act(x)
	   res = x
		
	   #L2/3
	   x = self.conv_L2_L2_3_keep(x)
	   x = act(x)
	   x = self.conv_L2_L2_3_keep(x) + res
	   x = act(x)
	   res = x
	   
	   #L2/4
	   x = self.conv_L2_L2_3_keep(x)
	   x = act(x)
	   x = self.conv_L2_L2_3_keep(x) + res
	   x = act(x)
	   	   
	   res = self.conv_L2_L3_1_down(x)
	   
	   #print('L3', res.size())
	   #L3/1
	   x = self.conv_L2_L3_3_down(x)
	   x = act(x)
	   x = self.conv_L3_L3_3_keep(x) + res
	   x = act(x)
	   res = x
	   
      #L3/2
	   x = self.conv_L3_L3_3_keep(x)
	   x = act(x)
	   x = self.conv_L3_L3_3_keep(x) + res
	   x = act(x)
	   res = x
	   
	   #L3/3
	   x = self.conv_L3_L3_3_keep(x)
	   x = act(x)
	   x = self.conv_L3_L3_3_keep(x) + res
	   x = act(x)
	   res = x
	   
	   #L3/4
	   x = self.conv_L3_L3_3_keep(x)
	   x = act(x)
	   x = self.conv_L3_L3_3_keep(x) + res
	   x = act(x)	   
	   
	   res = self.conv_L3_L4_1_down(x)
	   
	   #print('L4', res.size())
	   #L4/1
	   x = self.conv_L3_L4_3_down(x)
	   x = act(x)
	   x = self.conv_L4_L4_3_keep(x) + res
	   x = act(x)
	   res = x
	   
	   #L4/2
	   x = self.conv_L4_L4_3_keep(x)
	   x = act(x)
	   x = self.conv_L4_L4_3_keep(x) + res
	   x = act(x)
	   res = x
	   	   
	   #L4/3
	   x = self.conv_L4_L4_3_keep(x)
	   x = act(x)
	   x = self.conv_L4_L4_3_keep(x) + res
	   x = act(x)
	   res = x

	   #L4/4
	   x = self.conv_L4_L4_3_keep(x)
	   x = act(x)
	   x = self.conv_L4_L4_3_keep(x) + res
	   x = act(x)
	   
	   res = res = self.conv_L4_L5_1_down(x)
	   
	   #L5/1
	   x = self.conv_L4_L5_3_down(x)
	   x = act(x)
	   x = self.conv_L5_L5_3_keep(x) + res
	   x = act(x)
	   res = x
	   
	   #L5/2
	   x = self.conv_L5_L5_3_keep(x)
	   x = act(x)
	   x = self.conv_L5_L5_3_keep(x) + res
	   x = act(x)
	   res = x
	   	   
	   #L5/3
	   x = self.conv_L5_L5_3_keep(x)
	   x = act(x)
	   x = self.conv_L5_L5_3_keep(x) + res
	   x = act(x)
	   res = x

	   #L5/4
	   x = self.conv_L5_L5_3_keep(x)
	   x = act(x)
	   x = self.conv_L5_L5_3_keep(x) + res
	   x = act(x)
	   
	   #print('before pooling', x.size())
	   x = self.adaptive_pool2d(x)
	   #print('before flatten', x.size())
	   x = self.lin1(x)
	   return x
