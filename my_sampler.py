import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, MFCC
import pandas as pd
import os
import random
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import librosa
import lmdb
import msgpack
import msgpack_numpy as m

m.patch() # Патчим msgpack для поддержки numpy


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
	fig, axs = plt.subplots(1, 1)
	axs.set_title(title or 'Spectrogram (db)')
	axs.set_ylabel(ylabel)
	axs.set_xlabel('frame')
	im = axs.imshow(spec, origin='lower', aspect=aspect)
	if xmax:
		axs.set_xlim((0, xmax))
	fig.colorbar(im, ax=axs)
	plt.show(block=False)



class RandomSpeedChange:
	def __init__(self, sample_rate):
		self.sample_rate = sample_rate
	
	def __call__(self, audio_data):
		speed_factor = random.choice([0.9, 0.95, 1.05, 1.1])
		# change speed and resample to original rate
		transformed_audio = librosa.effects.time_stretch(y = audio_data, rate = speed_factor)
		return transformed_audio

class RandomBackgroundNoise:
	def __init__(self, sample_rate, min_snr_db=10, max_snr_db=15):
		self.sample_rate = sample_rate
		self.min_snr_db = min_snr_db
		self.max_snr_db = max_snr_db
		
		#if not os.path.exists(noise_dir):
		#	raise IOError(f'Noise directory `{noise_dir}` does not exist')
		# find all WAV files including in sub-folders:
		#self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))
		#if len(self.noise_files_list) == 0:
		#	raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')
	
	def __call__(self, main_audio, noise_audio):
		#noise_audio, _ = librosa.load(random.choice(self.noise_files_list), sr = self.sample_rate, mono = True)
		# 2. Приводим к ОДНОЙ длине (обрезаем или дополняем нулями более короткий сигнал)
		#  Выравниваем по длине основного трека
		if len(noise_audio) > len(main_audio):
			# Если наложение длиннее - обрезаем его
			noise_audio = noise_audio[:len(main_audio)]
		else:
			# Если наложение короче - дополняем нулями (паузами) до нужной длины
			padding = len(main_audio) - len(noise_audio)
			noise_audio = np.pad(noise_audio, (0, padding), mode='constant')
				
		# 3. Смешиваем с регулировкой громкости (чтобы не было клиппинга)
		# Можно задать весовые коэффициенты для каждого сигнала
		main_gain = 0.9   # Громкость основного трека (90%)
		noise_gain = 0.1 # Громкость накладываемого трека (10%)
		mixed_audio = (main_gain * main_audio) + (noise_gain * noise_audio)

		# 4. НОРМАЛИЗАЦИЯ: Предотвращаем возможное превышение амплитуды (>1.0 или <-1.0)
		# Это критически важно для сохранения без искажений
		if np.max(np.abs(mixed_audio)) > 1.0:
			mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
			
		return mixed_audio


    
class YesDataset(Dataset):
	def __init__(self, dataset_type, annotations_file, sound_dir, noise_annotation, noise_dir, lmdb_audio_path, lmdb_noise_path, map_size, sample_rate, sound_len, n_mels, n_mfcc, USE_AGUMENTATION, project, n_fft = 2048, hop_length = 512, transform = None, target_transform = None, feature = 'MelSpectrogram'):
		self.dataset_type = dataset_type
		self.sound_labels = pd.read_csv(annotations_file)
		self.sound_dir = sound_dir
		self.noise_files = pd.read_csv(noise_annotation)
		self.noise_dir = noise_dir
		self.lmdb_audio_path = lmdb_audio_path
		self.lmdb_noise_path = lmdb_noise_path
		self.map_size = map_size
		self.transform = transform
		self.target_transform = target_transform
		self.resample_rate = sample_rate
		self.tensor_size = sound_len * sample_rate
		self.USE_AGUMENTATION = USE_AGUMENTATION
		self.n_mfcc = n_mfcc
		self.n_mels = n_mels
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.feature = feature
		self.rate = sample_rate
		self.project = project

		self.audio_env = lmdb.open(self.lmdb_audio_path, map_size=map_size, readonly=False, lock=False)
		self.noise_env = lmdb.open(self.lmdb_noise_path, map_size=map_size, readonly=False, lock=False)

		# Заполняем базу если пустая
		self._fill_database()
	
	def _fill_database(self):
		if self.project == "Medley-solos-DB":
			self.sound_labels = self.sound_labels[self.sound_labels.iloc[:, 0] == self.dataset_type]
				
		with self.audio_env.begin(write=True) as txn:
			for i in range(len(self.sound_labels)):
				key = str(i).encode()
				if not txn.get(key):
					if self.project == "Medley-solos-DB":
						sound_path = os.path.join(self.sound_dir, "Medley-solos-DB_"+self.dataset_type+"-"+self.sound_labels.iloc[i, 2]+"_"+self.sound_labels.iloc[i, 4]+".wav")
					else:
						sound_path = os.path.join(self.sound_dir, self.sound_labels.iloc[i, 0])
					sound, sr = librosa.load(sound_path, sr = self.rate, mono = True)
					data = {'sound': sound.astype(np.float32), 'sample_rate': sr}
					txn.put(key, msgpack.packb(data))

		with self.noise_env.begin(write=True) as txn:
			for i in range(len(self.noise_files)):
				key = str(i).encode()
				if not txn.get(key):
					sound_path = os.path.join(self.noise_dir, self.noise_files.iloc[i, 0]+'.wav')
					sound, sr = librosa.load(sound_path, sr = self.rate, mono = True)
					data = {'sound': sound.astype(np.float32), 'sample_rate': sr}
					txn.put(key, msgpack.packb(data))

								
	def __len__(self):
		return len(self.sound_labels)
	
	def __getitem__(self, idx):
		with self.audio_env.begin() as txn:
			data = msgpack.unpackb(txn.get(str(idx).encode()))
			sound = data['sound']

		if self.project == "Medley-solos-DB":
			label = int(self.sound_labels.iloc[idx, 2])
		else:
			label = int(self.sound_labels.iloc[idx, 1])
		#print(self.sound_labels)
				
		# source sound agumentation
		if self.USE_AGUMENTATION:
			agumentation = random.choice(['nothing', 'speed', 'noise', 'speed + noise'])
			with self.noise_env.begin() as txn:
				idx = random.randint(0, len(self.noise_files)-1)
				data = msgpack.unpackb(txn.get(str(idx).encode()))
				noise = data['sound']
		else:
			agumentation = 'nothing'
		
		if agumentation == 'speed':
			speed_transform = RandomSpeedChange(self.resample_rate)
			sound = speed_transform(sound)
		elif agumentation == 'noise':
			noise_transform = RandomBackgroundNoise(self.resample_rate)
			sound = noise_transform(sound, noise)
		elif agumentation == 'speed + noise':
			speed_transform = RandomSpeedChange(self.resample_rate)
			sound = speed_transform(sound)
			noise_transform = RandomBackgroundNoise(self.resample_rate)
			sound = noise_transform(sound, noise)
			
		# apply lenght
		sound = torch.tensor(sound)
#		sound = sound.squeeze(0)
		if sound.size()[0]  <  self.tensor_size:
			sound = nn.ConstantPad1d((0, self.tensor_size - sound.size()[0]), 0)(sound)
		elif sound.size()[0]  >  self.tensor_size:
			sound = sound.narrow(0, 0, self.tensor_size)
		
		#print('sound', sound.shape)
		#uncomment if sound_length = 20 sound = nn.ConstantPad1d((0, self.hop_length), 0)(sound)
		#print('sound', sound.shape)
		
		
		#convert to spectrogram
		sound = sound.unsqueeze(0)
		
		if self.feature == 'MelSpectrogram':
			mspectre = MelSpectrogram(sample_rate=self.resample_rate, n_fft=self.n_fft, hop_length=self.hop_length, center=True, pad_mode="reflect", power=2.0, norm='slaney', n_mels=self.n_mels, mel_scale="htk")
			decibel = AmplitudeToDB(top_db=80)
			return_feature = decibel(mspectre(sound))
		elif self.feature == 'MFCC':
			mfcc_transform = MFCC(sample_rate=self.resample_rate, n_mfcc=self.n_mfcc, melkwargs={"n_fft": self.n_fft, "n_mels": self.n_mels, "hop_length": self.hop_length, "mel_scale": "htk"})
			return_feature = mfcc_transform(sound)
		
		pad_amount = 1024 - return_feature.size(dim=2)
		return_feature = F.pad(return_feature, (0, pad_amount))
				
		if self.transform:
			sound_nda = self.transform.fit_transform(return_feature)
			return_feature = torch.Tensor(sound_nda)		
		#return_feature = return_feature.unsqueeze(0)
		return return_feature, label
