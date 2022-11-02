import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio
from torchaudio.transforms import  MelSpectrogram, Resample, AmplitudeToDB
import random
import pathlib
import math
import matplotlib.pyplot as plt

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
	def __init__(self, annotations_file, sound_dir, sample_rate, sound_len, n_mels, USE_TRIPLE_SPECS=False, n_fft= 400, hop_length=160, transform=None, target_transform=None):
		self.sound_labels = pd.read_csv(annotations_file)
		self.sound_dir = sound_dir
		self.transform = transform
		self.target_transform = target_transform
		self.resample_rate=sample_rate
		self.tensor_size = sound_len * sample_rate
		self.USE_TRIPLE_SPECS = USE_TRIPLE_SPECS
		self.n_mels = n_mels
		self.n_fft = n_fft
		self.hop_length=hop_length		
				
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
		effects = [['remix', '1'], ['rate', str(self.resample_rate)],]
		sound, _ = torchaudio.sox_effects.apply_effects_tensor(sound, self.resample_rate, effects)
		
		# source sound agumentation
		agumentation = random.choice(['nothing', 'speed', 'noise', 'speed + noise'])
		#agumentation = random.choice(['none', 'speed'])
		#agumentation = random.choice(['nothing', 'noise'])
		#agumentation = 'none'
		if agumentation == 'speed':
			speed_transform = RandomSpeedChange(self.resample_rate)
			sound = speed_transform(sound)
		elif agumentation == 'noise':
			noise_transform = RandomBackgroundNoise(self.resample_rate, './data/noise')
			sound = noise_transform(sound)
		elif agumentation == 'speed + noise':
			speed_transform = RandomSpeedChange(self.resample_rate)
			sound = speed_transform(sound)
			noise_transform = RandomBackgroundNoise(self.resample_rate, './data/noise')
			sound = noise_transform(sound)
			
		# apply 20 sec lenght
		sound = sound.squeeze(0)
		if sound.size()[0]  <  self.tensor_size:
			sound = nn.ConstantPad1d((0, self.tensor_size - sound.size()[0] ), 0)(sound)
		elif sound.size()[0]  >  self.tensor_size:
			sound = sound.narrow(0, 0, self.tensor_size)
		
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
		if self.USE_TRIPLE_SPECS:
			mspectre1 = MelSpectrogram(sample_rate=self.resample_rate, n_fft=n_fft1, hop_length=hop_length1, center=True, pad_mode="reflect", power=2.0, onesided=True, n_mels=n_mels, mel_scale="htk")
			mspectre2 = MelSpectrogram(sample_rate=self.resample_rate, n_fft=n_fft2, hop_length=hop_length2, center=True, pad_mode="reflect", power=2.0, onesided=True, n_mels=n_mels, mel_scale="htk")
			mspectre3 = MelSpectrogram(sample_rate=self.resample_rate, n_fft=n_fft3, hop_length=hop_length3, center=True, pad_mode="reflect", power=2.0, onesided=True, n_mels=n_mels, mel_scale="htk")
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
			mspectre = MelSpectrogram(sample_rate=self.resample_rate, n_fft=self.n_fft, hop_length=self.hop_length, center=True, pad_mode="reflect", power=2.0, onesided=True, n_mels=self.n_mels, mel_scale="htk")
			sound = mspectre(sound)
			sound = decibel(sound)
			img = sound.squeeze(0)
			plt.imshow(img)
		return sound, label
