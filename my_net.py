import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


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
		#self.conv_1_L1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=L1, kernel_size=5, stride=2, padding=3, bias=False), nn.BatchNorm2d(self.L1))
		self.conv_1_L1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=L1, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L1))
		
		self.conv_L1_L2_1_down = nn.Sequential(nn.Conv2d(in_channels=self.L1, out_channels=self.L2, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L2))
		self.conv_L2_L3_1_down = nn.Sequential(nn.Conv2d(in_channels=self.L2, out_channels=self.L3, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L3))
		self.conv_L3_L4_1_down = nn.Sequential(nn.Conv2d(in_channels=self.L3, out_channels=self.L4, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L4))
		self.conv_L4_L5_1_down = nn.Sequential(nn.Conv2d(in_channels=self.L4, out_channels=self.L5, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L5))
		
		self.conv_L1_L2_3_down = nn.Sequential(nn.Conv2d(in_channels=self.L1, out_channels=self.L2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.L2))
		self.conv_L2_L3_3_down = nn.Sequential(nn.Conv2d(in_channels=self.L2, out_channels=self.L3, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.L3))
		self.conv_L3_L4_3_down = nn.Sequential(nn.Conv2d(in_channels=self.L3, out_channels=self.L4, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.L4))
		self.conv_L4_L5_3_down = nn.Sequential(nn.Conv2d(in_channels=self.L4, out_channels=self.L5, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(self.L5))
				
		self.conv_L1_L1_3_keep = nn.Sequential(nn.Conv2d(in_channels=self.L1, out_channels=self.L1, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L1))
		self.conv_L2_L2_3_keep = nn.Sequential(nn.Conv2d(in_channels=self.L2, out_channels=self.L2, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L2))
		self.conv_L3_L3_3_keep = nn.Sequential(nn.Conv2d(in_channels=self.L3, out_channels=self.L3, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L3))
		self.conv_L4_L4_3_keep = nn.Sequential(nn.Conv2d(in_channels=self.L4, out_channels=self.L4, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L4))
		self.conv_L5_L5_3_keep = nn.Sequential(nn.Conv2d(in_channels=self.L5, out_channels=self.L5, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L5))
				
		self.adaptive_pool2d = nn.AdaptiveAvgPool2d((2, 2))
		self.lin1 = nn.Sequential(nn.Flatten(), nn.Linear(self.inner_dim, 1024))
		self.lin2 = nn.Sequential(nn.Linear(1024, self.embedding_dim))
		self.flatten = nn.Flatten()
		self.lin = nn.Linear(self.inner_dim*2*2, self.embedding_dim)
		#self.apply(weights_init_kaiming)
		#self.apply(fc_init_weights)

	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
				nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
	
	def forward(self, x):
		#act = nn.LeakyReLU(0.1)
		act = nn.ReLU()
		#print('input', x.size())
		x = self.conv_1_L1(x)
		x = act(x)
		#x = F.max_pool2d(x, 2)
		res = x   
		#print('L1', x.size())
		#L1/1
		x = act(self.conv_L1_L1_3_keep(x))
		x = act(self.conv_L1_L1_3_keep(x)) + res
		res = x
	   
		#L1/2
		x = act(self.conv_L1_L1_3_keep(x))
		x = act(self.conv_L1_L1_3_keep(x)) + res
		res = x
	   
		#L1/3
		x = act(self.conv_L1_L1_3_keep(x))
		x = act(self.conv_L1_L1_3_keep(x)) + res
		res = x

		#L1/4
		x = act(self.conv_L1_L1_3_keep(x))
		x = act(self.conv_L1_L1_3_keep(x)) + res
	   	   	   
		res = self.conv_L1_L2_3_down(x)

		#print('L2', res.size())
		#L2/1
		x = act(self.conv_L1_L2_3_down(x))
		x = act(self.conv_L2_L2_3_keep(x)) + res
		res = x
	   
		#L2/2
		x = act(self.conv_L2_L2_3_keep(x))
		x = act(self.conv_L2_L2_3_keep(x)) + res
		res = x
		
		#L2/3
		x = act(self.conv_L2_L2_3_keep(x))
		x = act(self.conv_L2_L2_3_keep(x)) + res
		res = x
	   
		#L2/4
		x = act(self.conv_L2_L2_3_keep(x))
		x = act(self.conv_L2_L2_3_keep(x)) + res
	      	   
		# L3
		res = self.conv_L2_L3_3_down(x)
	   
		#print('L3', res.size())
		#L3/1
		x = act(self.conv_L2_L3_3_down(x))
		x = act(self.conv_L3_L3_3_keep(x)) + res
		res = x
	   
		#L3/2
		x = act(self.conv_L3_L3_3_keep(x))
		x = act(self.conv_L3_L3_3_keep(x)) + res
		res = x
	   
		#L3/3
		x = act(self.conv_L3_L3_3_keep(x))
		x = act(self.conv_L3_L3_3_keep(x)) + res
		res = x
	   
		#L3/4
		x = act(self.conv_L3_L3_3_keep(x))
		x = act(self.conv_L3_L3_3_keep(x)) + res
	   	   
		#L4
		res = self.conv_L3_L4_3_down(x)
	   
		#print('L4', res.size())
		
		#L4/1
		x = act(self.conv_L3_L4_3_down(x))
		x = act(self.conv_L4_L4_3_keep(x)) + res
		res = x
	   
		#L4/2
		x = act(self.conv_L4_L4_3_keep(x))
		x = act(self.conv_L4_L4_3_keep(x)) + res
		res = x
	   	   
		#L4/3
		x = act(self.conv_L4_L4_3_keep(x))
		x = act(self.conv_L4_L4_3_keep(x)) + res
		res = x

		#L4/4
		x = act(self.conv_L4_L4_3_keep(x))
		x = act(self.conv_L4_L4_3_keep(x)) + res
	   	   
		#L5
		res = self.conv_L4_L5_3_down(x)

		#print('L4', res.size())
	   
		#L5/1
		x = act(self.conv_L4_L5_3_down(x))
		x = act(self.conv_L5_L5_3_keep(x)) + res
		res = x

		#L5/2
		x = act(self.conv_L5_L5_3_keep(x))
		x = act(self.conv_L5_L5_3_keep(x)) + res
		res = x

		#L5/3
		x = act(self.conv_L5_L5_3_keep(x))
		x = act(self.conv_L5_L5_3_keep(x)) + res
		res = x

		#L5/4
		x = act(self.conv_L5_L5_3_keep(x))
		x = act(self.conv_L5_L5_3_keep(x)) + res

		#print('before pooling', x.size())
		x = self.adaptive_pool2d(x)
		#print('before flatten', x.size())
		x = self.flatten(x)
		#print('after flatten', x.size())
		x = self.lin(x)
		#print('result', x.size())
		return x

# Autoencoder approach
class VariationalEncoder(nn.Module):
	def __init__(self, L1, L2, L3, L4, inner_dim, embedding_dim):
		super(VariationalEncoder, self).__init__()
		self.L1 = L1
		self.L2 = L2
		self.L3 = L3
		self.L4 = L4
		self.inner_dim = inner_dim
		self.embedding_dim = embedding_dim
		self.conv_1_L1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=L1, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(self.L1))
		#self.conv_1_L1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=L1, kernel_size=3, stride=2, padding = 1, bias=False), nn.BatchNorm2d(self.L1))
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
		self.drop = nn.Dropout(0.1)
		
		self.pool2d = nn.MaxPool2d(2, return_indices=True)
		self.lin1 = nn.Sequential(nn.Flatten(), nn.Linear(self.inner_dim, 1024))
		self.lin2 = nn.Sequential(nn.Linear(1024, self.embedding_dim))
		self.flatten = nn.Flatten()
		self.lin = nn.Linear(self.inner_dim*2*2, self.embedding_dim)

		
		# Adding mean and sigma projections
		self.mu = nn.Linear(self.inner_dim, self.embedding_dim)
		self.sigma = nn.Linear(self.inner_dim, self.embedding_dim)

		self.N = torch.distributions.Normal(0, 1)
		self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
		self.N.scale = self.N.scale.cuda()
		self.kl = 0

		#self.weight_init()
		
	def weight_init(self):
		for block in self._modules:
			for m in self._modules[block]:
				kaiming_init(m)
	
	
	def forward(self, x):
		#act = nn.LeakyReLU(0.1)
		#act = nn.ReLU()
		act = nn.LeakyReLU(0.2)
		#print('input', x.size())

		#print("Mean:", torch.mean(x))
		#print("Standard deviation:", torch.std(x))
		#print("Min:", torch.min(x))
		#print("Max:", torch.max(x))
		
		x = self.conv_1_L1(x)
		x = act(x)
		#x, pool_ind1 = F.max_pool2d(x, 2, return_indices=True)
		#print('after 1st pooling', x.size())
		res = x
		
		#print('L1', x.size())
		#L1/1
		x = act(self.conv_L1_L1_3_keep(x))
		x = act(self.conv_L1_L1_3_keep(x)) + res
		res = x
		
		#L1/2
		x = act(self.conv_L1_L1_3_keep(x))
		x = act(self.conv_L1_L1_3_keep(x)) + res
		res = x
		
		#L1/3
		x = act(self.conv_L1_L1_3_keep(x))
		x = act(self.conv_L1_L1_3_keep(x)) + res
		res = x
		
		#L1/4
		x = act(self.conv_L1_L1_3_keep(x))
		x = act(self.conv_L1_L1_3_keep(x)) + res
		res = self.conv_L1_L2_3_down(x)
		
		#print('L2', res.size())
		#L2/1
		x = act(self.conv_L1_L2_3_down(x))
		x = act(self.conv_L2_L2_3_keep(x)) + res
		res = x
		
		#L2/2
		x = act(self.conv_L2_L2_3_keep(x))
		x = act(self.conv_L2_L2_3_keep(x)) + res
		res = x

		#L2/3
		x = act(self.conv_L2_L2_3_keep(x))
		x = act(self.conv_L2_L2_3_keep(x)) + res
		res = x

		#L2/4
		x = act(self.conv_L2_L2_3_keep(x))
		x = act(self.conv_L2_L2_3_keep(x)) + res
		res = self.conv_L2_L3_3_down(x)
		
		#print('L3', res.size())
		#L3/1
		x = act(self.conv_L2_L3_3_down(x))
		x = act(self.conv_L3_L3_3_keep(x)) + res
		res = x
		
		#L3/2
		x = act(self.conv_L3_L3_3_keep(x))
		x = act(self.conv_L3_L3_3_keep(x)) + res
		res = x

		#L3/3
		x = act(self.conv_L3_L3_3_keep(x))
		x = act(self.conv_L3_L3_3_keep(x)) + res
		res = x

		#L3/4
		x = act(self.conv_L3_L3_3_keep(x))
		x = act(self.conv_L3_L3_3_keep(x)) + res
		res = self.conv_L3_L4_3_down(x)
		
		#print('L4', res.size())
		#L4/1
		x = act(self.conv_L3_L4_3_down(x))
		x = act(self.conv_L4_L4_3_keep(x)) + res
		res = x
		
		#L4/2
		x = act(self.conv_L4_L4_3_keep(x))
		x = act(self.conv_L4_L4_3_keep(x)) + res
		res = x

		#L4/3
		x = act(self.conv_L4_L4_3_keep(x))
		x = act(self.conv_L4_L4_3_keep(x)) + res
		res = x

		#L4/4
		x = act(self.conv_L4_L4_3_keep(x))
		x = act(self.conv_L4_L4_3_keep(x)) + res
				
		#x = self.drop(x)
		#print('before pooling', x.size())
		
		#FInal pooling
		x, pool_ind = self.pool2d(x)
		#x, pool_ind = F.max_pool2d(x, 4, return_indices=True)
		#print('after pooling', x.size())
		
		#x = act(torch.flatten(x, start_dim=1))
		x = torch.flatten(x, start_dim=1)
		#print('after flatten', x.size())
		
		#self.kl = torch.zeros(x.size(), dtype=torch.float64, device='cuda:0')
		# getting mean/sigma projections
		x_mu = self.mu(x)
		x_sigma = torch.exp(self.sigma(x))
		# reparameterization trick
		z = x_mu + x_sigma * self.N.sample(x_mu.shape)
	
		# compute the KL divergence and store in the class
		self.kl = (x_sigma ** 2 + x_mu ** 2 - torch.log(x_sigma) - 0.5).sum()

		return z, pool_ind
        
class Decoder(nn.Module):
    def __init__(self, L1, L2, L3, L4, inner_dim, embedding_dim, pool_ind=0):
        super().__init__()
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.inner_dim = inner_dim
        self.embedding_dim = embedding_dim
        self.pool_ind = pool_ind
                
        # Linear section
        self.decoder_lin = nn.Linear(self.embedding_dim, self.inner_dim)

        # Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.L4, 8, 8))
        
        # Unpooling
        self.unpool = nn.MaxUnpool2d(2)
        self.unpool2_D = nn.MaxUnpool2d(2, stride=2)
        
        self.drop = nn.Dropout(0.1)
        
        # ConvTranspose
        self.conv_L1_1_D = nn.Sequential(nn.ConvTranspose2d(in_channels=L1, out_channels=1, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False), nn.BatchNorm2d(1))
        self.conv_L1_L1_3_keep_D = nn.Sequential(nn.Conv2d(in_channels=self.L1, out_channels=self.L1, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L1))
        self.conv_L2_L1_1_up_D = nn.Sequential(nn.ConvTranspose2d(in_channels=self.L2, out_channels=self.L1, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L1))
        self.conv_L2_L1_3_up_D = nn.Sequential(nn.ConvTranspose2d(in_channels=self.L2, out_channels=self.L1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(self.L1))
        self.conv_L2_L2_3_keep_D = nn.Sequential(nn.Conv2d(in_channels=self.L2, out_channels=self.L2, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L2))
        self.conv_L3_L2_1_up_D = nn.Sequential(nn.ConvTranspose2d(in_channels=self.L3, out_channels=self.L2, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L2))
        self.conv_L3_L2_3_up_D = nn.Sequential(nn.ConvTranspose2d(in_channels=self.L3, out_channels=self.L2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(self.L2))
        self.conv_L3_L3_3_keep_D = nn.Sequential(nn.Conv2d(in_channels=self.L3, out_channels=self.L3, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L3))
        self.conv_L4_L3_1_up_D = nn.Sequential(nn.ConvTranspose2d(in_channels=self.L4, out_channels=self.L3, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L3))
        self.conv_L4_L3_3_up_D = nn.Sequential(nn.ConvTranspose2d(in_channels=self.L4, out_channels=self.L3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(self.L3))
        self.conv_L4_L4_3_keep_D = nn.Sequential(nn.Conv2d(in_channels=self.L4, out_channels=self.L4, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L4))
        
    def forward(self, x):
        #act = nn.LeakyReLU(0.1)
        #act = nn.ReLU()
        act = nn.LeakyReLU(0.2)
        
        # Apply linear layer
        x = self.decoder_lin(x)
        #print('fully connected', x.size())
        # Unflatten
        x = self.unflatten(x)
        #print('after unflatten', x.size())
        
        #unpool 
        x = self.unpool(x, self.pool_ind)
        #print('after 1st unpool', x.size())
 
        # Apply transposed convolutions
        #L4
        res = x
        x = act(self.conv_L4_L4_3_keep_D(x))
        x = act(self.conv_L4_L4_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L4_L4_3_keep_D(x))
        x = act(self.conv_L4_L4_3_keep_D(x)) + res
		
        res = x
        x = act(self.conv_L4_L4_3_keep_D(x))
        x = act(self.conv_L4_L4_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L4_L4_3_keep_D(x))
        x = act(self.conv_L4_L4_3_keep_D(x)) + res

        #L3
        res = self.conv_L4_L3_3_up_D(x)
        x = act(self.conv_L4_L3_3_up_D(x))
        x = act(self.conv_L3_L3_3_keep_D(x)) + res
        
        res = x
        x = act(self.conv_L3_L3_3_keep_D(x))
        x = act(self.conv_L3_L3_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L3_L3_3_keep_D(x))
        x = act(self.conv_L3_L3_3_keep_D(x)) + res
        
        res = x
        x = act(self.conv_L3_L3_3_keep_D(x))
        x = act(self.conv_L3_L3_3_keep_D(x)) + res
        
        #L2
        res = self.conv_L3_L2_3_up_D(x)
        x = act(self.conv_L3_L2_3_up_D(x))
        x = act(self.conv_L2_L2_3_keep_D(x)) + res
        
        res = x
        x = act(self.conv_L2_L2_3_keep_D(x))
        x = act(self.conv_L2_L2_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L2_L2_3_keep_D(x))
        x = act(self.conv_L2_L2_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L2_L2_3_keep_D(x))
        x = act(self.conv_L2_L2_3_keep_D(x)) + res

        #L1
        res = self.conv_L2_L1_3_up_D(x)
        x = act(self.conv_L2_L1_3_up_D(x))
        x = act(self.conv_L1_L1_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L1_L1_3_keep_D(x))
        x = act(self.conv_L1_L1_3_keep_D(x)) + res
         
        res = x
        x = act(self.conv_L1_L1_3_keep_D(x))
        x = act(self.conv_L1_L1_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L1_L1_3_keep_D(x))
        x = act(self.conv_L1_L1_3_keep_D(x)) + res
        
		#Restore to initial size
        x = self.conv_L1_1_D(x)
                        
        #x = torch.sigmoid(x)
        return x
        
class VariationalAutoencoder(nn.Module):
    def __init__(self, L1, L2, L3, L4, inner_dim, embedding_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(L1, L2, L3, L4, inner_dim, embedding_dim)
        self.decoder = Decoder(L1, L2, L3, L4, inner_dim, embedding_dim)
        
    def forward(self, x):
        z, ind = self.encoder(x)
        self.decoder.pool_ind = ind
        return self.decoder(z)
	

### Alternative approach
class VAE_Encoder(nn.Module):
	def __init__(self, L1, L2, L3, L4, inner_dim, embedding_dim):
		super(VAE_Encoder, self).__init__()
		self.L1 = L1
		self.L2 = L2
		self.L3 = L3
		self.L4 = L4
		self.inner_dim = inner_dim
		self.embedding_dim = embedding_dim
		self.conv_1_L1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=L1, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(self.L1))
		#self.conv_1_L1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=L1, kernel_size=3, stride=2, padding = 1, bias=False), nn.BatchNorm2d(self.L1))
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
		self.drop = nn.Dropout(0.1)
		
		self.pool2d = nn.MaxPool2d(2, return_indices=True)
		self.lin1 = nn.Sequential(nn.Flatten(), nn.Linear(self.inner_dim, 1024))
		self.lin2 = nn.Sequential(nn.Linear(1024, self.embedding_dim))
		self.flatten = nn.Flatten()
		self.lin = nn.Linear(self.inner_dim*2*2, self.embedding_dim)

		
		# Adding mean and sigma projections
		self.mu = nn.Linear(self.inner_dim, self.embedding_dim)
		self.sigma = nn.Linear(self.inner_dim, self.embedding_dim)

		self.N = torch.distributions.Normal(0, 1)
		self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
		self.N.scale = self.N.scale.cuda()
		self.kl = 0

		#self.weight_init()
		
	def weight_init(self):
		for block in self._modules:
			for m in self._modules[block]:
				kaiming_init(m)
	
	
	def forward(self, x):
		#act = nn.ReLU()
		act = nn.LeakyReLU(0.2)
		
		x = self.conv_1_L1(x)
		x = act(x)
		#x, pool_ind1 = F.max_pool2d(x, 2, return_indices=True)
		#print('after 1st pooling', x.size())
		res = x
		
		#print('L1', x.size())
		#L1/1
		x = act(self.conv_L1_L1_3_keep(x))
		x = act(self.conv_L1_L1_3_keep(x)) + res
		res = x
		
		#L1/2
		x = act(self.conv_L1_L1_3_keep(x))
		x = act(self.conv_L1_L1_3_keep(x)) + res
		res = x
		
		#L1/3
		x = act(self.conv_L1_L1_3_keep(x))
		x = act(self.conv_L1_L1_3_keep(x)) + res
		res = x
		
		#L1/4
		x = act(self.conv_L1_L1_3_keep(x))
		x = act(self.conv_L1_L1_3_keep(x)) + res
		res = self.conv_L1_L2_3_down(x)
		
		#print('L2', res.size())
		#L2/1
		x = act(self.conv_L1_L2_3_down(x))
		x = act(self.conv_L2_L2_3_keep(x)) + res
		res = x
		
		#L2/2
		x = act(self.conv_L2_L2_3_keep(x))
		x = act(self.conv_L2_L2_3_keep(x)) + res
		res = x

		#L2/3
		x = act(self.conv_L2_L2_3_keep(x))
		x = act(self.conv_L2_L2_3_keep(x)) + res
		res = x

		#L2/4
		x = act(self.conv_L2_L2_3_keep(x))
		x = act(self.conv_L2_L2_3_keep(x)) + res
		res = self.conv_L2_L3_3_down(x)
		
		#print('L3', res.size())
		#L3/1
		x = act(self.conv_L2_L3_3_down(x))
		x = act(self.conv_L3_L3_3_keep(x)) + res
		res = x
		
		#L3/2
		x = act(self.conv_L3_L3_3_keep(x))
		x = act(self.conv_L3_L3_3_keep(x)) + res
		res = x

		#L3/3
		x = act(self.conv_L3_L3_3_keep(x))
		x = act(self.conv_L3_L3_3_keep(x)) + res
		res = x

		#L3/4
		x = act(self.conv_L3_L3_3_keep(x))
		x = act(self.conv_L3_L3_3_keep(x)) + res
		res = self.conv_L3_L4_3_down(x)
		
		#print('L4', res.size())
		#L4/1
		x = act(self.conv_L3_L4_3_down(x))
		x = act(self.conv_L4_L4_3_keep(x)) + res
		res = x
		
		#L4/2
		x = act(self.conv_L4_L4_3_keep(x))
		x = act(self.conv_L4_L4_3_keep(x)) + res
		res = x

		#L4/3
		x = act(self.conv_L4_L4_3_keep(x))
		x = act(self.conv_L4_L4_3_keep(x)) + res
		res = x

		#L4/4
		x = act(self.conv_L4_L4_3_keep(x))
		x = act(self.conv_L4_L4_3_keep(x)) + res
				
		#x = self.drop(x)
		#print('before pooling', x.size())
		
		#FInal pooling
		x, pool_ind = self.pool2d(x)
		#x, pool_ind = F.max_pool2d(x, 4, return_indices=True)
		#print('after pooling', x.size())
		
		#x = act(torch.flatten(x, start_dim=1))
		x = torch.flatten(x, start_dim=1)
		#print('after flatten', x.size())
		
		# getting mean/sigma projections
		x_mean = self.mu(x)
		x_logvar = self.sigma(x)
		
		return x_mean, x_logvar, pool_ind
	
class VAE_Decoder(nn.Module):
    def __init__(self, L1, L2, L3, L4, inner_dim, embedding_dim):
        super().__init__()
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.inner_dim = inner_dim
        self.embedding_dim = embedding_dim
                        
        # Linear section
        self.decoder_lin = nn.Linear(self.embedding_dim, self.inner_dim)

        # Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.L4, 8, 8))
        
        # Unpooling
        self.unpool = nn.MaxUnpool2d(2)
        self.unpool2_D = nn.MaxUnpool2d(2, stride=2)
        
        self.drop = nn.Dropout(0.1)
        
        # ConvTranspose
        self.conv_L1_1_D = nn.Sequential(nn.ConvTranspose2d(in_channels=L1, out_channels=1, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False), nn.BatchNorm2d(1))
        self.conv_L1_L1_3_keep_D = nn.Sequential(nn.Conv2d(in_channels=self.L1, out_channels=self.L1, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L1))
        self.conv_L2_L1_1_up_D = nn.Sequential(nn.ConvTranspose2d(in_channels=self.L2, out_channels=self.L1, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L1))
        self.conv_L2_L1_3_up_D = nn.Sequential(nn.ConvTranspose2d(in_channels=self.L2, out_channels=self.L1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(self.L1))
        self.conv_L2_L2_3_keep_D = nn.Sequential(nn.Conv2d(in_channels=self.L2, out_channels=self.L2, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L2))
        self.conv_L3_L2_1_up_D = nn.Sequential(nn.ConvTranspose2d(in_channels=self.L3, out_channels=self.L2, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L2))
        self.conv_L3_L2_3_up_D = nn.Sequential(nn.ConvTranspose2d(in_channels=self.L3, out_channels=self.L2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(self.L2))
        self.conv_L3_L3_3_keep_D = nn.Sequential(nn.Conv2d(in_channels=self.L3, out_channels=self.L3, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L3))
        self.conv_L4_L3_1_up_D = nn.Sequential(nn.ConvTranspose2d(in_channels=self.L4, out_channels=self.L3, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(self.L3))
        self.conv_L4_L3_3_up_D = nn.Sequential(nn.ConvTranspose2d(in_channels=self.L4, out_channels=self.L3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(self.L3))
        self.conv_L4_L4_3_keep_D = nn.Sequential(nn.Conv2d(in_channels=self.L4, out_channels=self.L4, kernel_size=3, stride=1, padding='same', bias=False), nn.BatchNorm2d(self.L4))
        
    def forward(self, x, pool_ind):
        #act = nn.LeakyReLU(0.1)
        #act = nn.ReLU()
        act = nn.LeakyReLU(0.2)
        
        # Apply linear layer
        x = self.decoder_lin(x)
        #print('fully connected', x.size())
        # Unflatten
        x = self.unflatten(x)
        #print('after unflatten', x.size())
        
        #unpool 
        x = self.unpool(x, pool_ind)
        #print('after 1st unpool', x.size())
 
        # Apply transposed convolutions
        #L4
        res = x
        x = act(self.conv_L4_L4_3_keep_D(x))
        x = act(self.conv_L4_L4_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L4_L4_3_keep_D(x))
        x = act(self.conv_L4_L4_3_keep_D(x)) + res
		
        res = x
        x = act(self.conv_L4_L4_3_keep_D(x))
        x = act(self.conv_L4_L4_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L4_L4_3_keep_D(x))
        x = act(self.conv_L4_L4_3_keep_D(x)) + res

        #L3
        res = self.conv_L4_L3_3_up_D(x)
        x = act(self.conv_L4_L3_3_up_D(x))
        x = act(self.conv_L3_L3_3_keep_D(x)) + res
        
        res = x
        x = act(self.conv_L3_L3_3_keep_D(x))
        x = act(self.conv_L3_L3_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L3_L3_3_keep_D(x))
        x = act(self.conv_L3_L3_3_keep_D(x)) + res
        
        res = x
        x = act(self.conv_L3_L3_3_keep_D(x))
        x = act(self.conv_L3_L3_3_keep_D(x)) + res
        
        #L2
        res = self.conv_L3_L2_3_up_D(x)
        x = act(self.conv_L3_L2_3_up_D(x))
        x = act(self.conv_L2_L2_3_keep_D(x)) + res
        
        res = x
        x = act(self.conv_L2_L2_3_keep_D(x))
        x = act(self.conv_L2_L2_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L2_L2_3_keep_D(x))
        x = act(self.conv_L2_L2_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L2_L2_3_keep_D(x))
        x = act(self.conv_L2_L2_3_keep_D(x)) + res

        #L1
        res = self.conv_L2_L1_3_up_D(x)
        x = act(self.conv_L2_L1_3_up_D(x))
        x = act(self.conv_L1_L1_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L1_L1_3_keep_D(x))
        x = act(self.conv_L1_L1_3_keep_D(x)) + res
         
        res = x
        x = act(self.conv_L1_L1_3_keep_D(x))
        x = act(self.conv_L1_L1_3_keep_D(x)) + res

        res = x
        x = act(self.conv_L1_L1_3_keep_D(x))
        x = act(self.conv_L1_L1_3_keep_D(x)) + res
        
		#Restore to initial size
        x = self.conv_L1_1_D(x)
                        
        x = torch.sigmoid(x)
        return x

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).cuda()        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var, pool_ind = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z, pool_ind)
        return x_hat, mean, log_var
