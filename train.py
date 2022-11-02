import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
from pytorch_metric_learning import losses, miners, distances, reducers, regularizers
#from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torchaudio
from nnAudio import Spectrogram
from prettytable import PrettyTable
import my_net as mn
import my_sampler as ms

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

inner_dim = 512
embedding_dim = 2
sound_len = 20
resample_rate = 14000
target_tensor_size = 544

n_mels = 128
n_fft1 = 400
n_fft2 = 800
n_fft3 = 1600
hop_length1 = 160
hop_length2 = 400
hop_length3 = 800

w_factor = 4
L1 = int(8 * w_factor)
L2 = int(16 * w_factor)
L3 = int(32 * w_factor)
L4 = int(64 * w_factor)
L5 = int(128 * w_factor)

classes = [
    "Ochi Chernye",
    "For Whom The Bell Tolls",
    "Fur Elise",
    "Polonaise",
    "Dont worry be happy",
    "I Was Made for Lovin You",
    "The House of the Rising Sun",
    "I Feel Good",
    "New York",
    "Smoke on the Water",
    "Highway to Hell",
    "Careless Whisper",
    "I Will Survive",
    "Let It Be",
    "Master Of Puppets",
    "Master Of Puppets_pt",
    "We Will Rock You",
    "Woman In Love",
    "Woman In Love_pt"
    ]
correct_answers = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
   
#L2 distance from point to set of points
def calc_l2_distance(a, X):
	#print('a:', a.size())
	#print('X:', X.size())
	z = X - a.repeat(X.size()[0], 1)
	zt = torch.zeros(X.size()[0],  embedding_dim, dtype = torch.float64)
	#y = torch.addcmul(zt, z, z)
	return torch.sqrt(torch.sum(torch.addcmul(zt, z, z), dim = 1))

#L2 distance between two points
def calc_l2(a, b):
	z = a - b
	zt = torch.zeros(a.size()[0], embedding_dim)
	return torch.sqrt(torch.sum(torch.addcmul(zt, z, z)))

def train(dataloader, model, loss_fn, optimizer, k):
	size = len(dataloader.dataset)
	#torch.backends.cudnn.benchmark = True
	batch_iteration = 0
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		batch_iteration += 1
		X, y = X.to(device), y.to(device)
		# Compute prediction error
		#optimizer.zero_grad()
		model.zero_grad(set_to_none = True)
		start_time = time.time()
		#with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
		embeddings = model(X)
		hard_pairs = miner(embeddings, y)
		loss = loss_fn(embeddings, y, hard_pairs)
						
		# Backpropagation
		#scaler.scale(loss).backward()
		loss.backward()
		#plot_grad_flow(model.named_parameters())
		
		#scaler.step(optimizer)
		optimizer.step()
		#scaler.update()
				
		#statistics
		if size % (batch_size*(batch_iteration+1)) != 0:
			loss, current = loss.item(), (batch+1) * len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] mined triplets: {miner.num_triplets}")
			#if use_scheduling: scheduler.step()
			
			#quickly estimate learinnig rate
			#compute predictions for train data by classes (K-nearest algorithm)
			correct_predictions = 0
			cpu_embeddings = embeddings.cpu()
			cpu_embeddings = cpu_embeddings.type(torch.float64)
			cpu_y = y.cpu()
			#cpu_embeddings = cpu_embeddings.cpu()
			#print('cpu_embeddings is cuda:', cpu_embeddings.is_cuda)
			for j in range(batch_size):
				test_embedding = cpu_embeddings[j]
				z, i = torch.sort(calc_l2_distance(test_embedding, cpu_embeddings), descending=False)
				s_labels = torch.index_select(cpu_y, 0, i[:k])
				freq = torch.bincount(s_labels)
				pred = torch.argmax(freq)
				if pred.item() == cpu_y[j]: correct_predictions += 1
			print('Correct predictions:', correct_predictions, end = ' ')
			print('out of:', batch_size)
			print('Batch accuracy:', "{:.2%}".format(correct_predictions/batch_size))

#-------------------------------------------------
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and (p.grad is not None):
            layers.append(n)
            pp = p.grad.abs().mean().item()
            ave_grads.append(pp)
    plt.figure()
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.show()
    
    #plt.grid(True)
    

def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
 
#----------------------------------------------
def test(dataloader, test_dataloader, model, k, embeddings):
	testing_size = len(test_dataloader.dataset)
#compute embbeddings for training data
	model.eval()
#	embeddings = torch.empty((0))
	labels = torch.empty((0), dtype = torch.int)
	
	for batch, (X, y) in enumerate(dataloader):
		X, batch_labels = X.to(device), y.to(device)
		labels = torch.cat((labels, batch_labels), 0)
	
#compute predictions for test data by classes (K-nearest algorithm)
	correct_predictions = 0
	for batch, (a, b) in enumerate(test_dataloader):
		a, b = a.to(device), b.to(device)
		model.eval()
		with torch.no_grad():
			test_embedding = model(a)
		z, i = torch.sort(calc_l2_distance(test_embedding, embeddings), descending=False)
		s_labels = torch.index_select(labels, 0, i[:k])
		freq = torch.bincount(s_labels)
		pred = torch.argmax(freq)
		correct_answers[b][1] += 1
		if pred.item() == b:
			correct_predictions += 1
			correct_answers[b][0] += 1
	print('Correct predictions:', correct_predictions, end = ' ')
	print('out of:', testing_size)
	print('Overall accuracy:', "{:.2%}".format(correct_predictions/testing_size))

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


#torch.set_num_threads(1)
# Create data loaders.
training_data = ms.YesDataset('/home/mkr/dl3/data/train/labels.csv', '/home/mkr/dl3/data/train', sample_rate = resample_rate, sound_len=sound_len, n_mels = n_mels)
test_data = ms.YesDataset('/home/mkr/dl3/data/validate/labels.csv', '/home/mkr/dl3/data/validate', sample_rate = resample_rate, sound_len=sound_len, n_mels = n_mels)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle = True, drop_last = True)
test_dataloader = DataLoader(training_data, batch_size=test_batch_size)
eval_dataloader = DataLoader(test_data, batch_size=test_batch_size)

train_len = training_data.__len__()
print('train_len:', train_len)

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#init the model
model = mn.wrn_nn(L1=L1, L2=L2, L3=L3, L4=L4, L5=L5, inner_dim=inner_dim, embedding_dim=embedding_dim).to(device)
#model = model.to(memory_format=torch.channels_last)
#model.init_weights()
#model.load_state_dict(torch.load("model-full-w10-50.pth"))
#model.apply(init_weights)
count_parameters(model)

#scaler = torch.cuda.amp.GradScaler()

### Pytorch-metric-learning setup ###
# Standard Triple Margin Loss
#distance = distances.LpDistance()
distance = distances.LpDistance(p=2, power=1, normalize_embeddings=False)
#distance = distances.SNRDistance()
#distance = distances.CosineSimilarity()
#reducer = reducers.ThresholdReducer(low = 0)
reducer = reducers.MeanReducer()
regularizer = regularizers.LpRegularizer()
loss_fn = losses.TripletMarginLoss(margin = 0.2, distance = distance, reducer = reducer, embedding_regularizer = regularizer)
#main_loss_fn = losses.TupletMarginLoss()
#var_loss_fn = losses.IntraPairVarianceLoss()
#loss_fn = losses.MultipleLosses([main_loss_fn, var_loss_fn], weights=[1, 0.5])
miner = miners.TripletMarginMiner(margin = 0.2, distance = distance, type_of_triplets = "semihard")
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay)

if use_scheduling:
	#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = min_lr, max_lr = max_lr, step_size_up = step_size_up, cycle_momentum = False, verbose =False)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = scheduler_step_size, gamma = scheduler_gamma)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones = [10,20], gamma = 0.1)

#accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)

print('Lets get ready for the rumble with', epochs, end = ' ' )
print('epochs!')

for t in range(epochs):
    print(' ')
    print(f"Epoch {t+1}\n-------------------------------")
    train_embeddings = train(train_dataloader, model, loss_fn, optimizer, k=knear)
    scheduler.step
    #print("Train accuracy")
    #test(train_dataloader, test_dataloader, model, knear, train_embeddings)
print("Eval accuracy")
#test(test_dataloader, eval_dataloader, model, knear)


print("Done!")

torch.save(model.state_dict(), "model-full-w6-ed2-100.pth")
print("Saved PyTorch Model State to model-full-w6-ed2-100.pth")
