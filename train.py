import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
#from torchvision.transforms import ToTensor, Lambda, Compose
from pytorch_metric_learning import losses, miners, distances, reducers, regularizers
#from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
#from nnAudio import Spectrogram
from prettytable import PrettyTable
import my_net as mn
import my_sampler as ms

#-- Константы для работы кастомного датасета по фрагментам песен
#PROJECT = 'melodyst'
#ANNOTATIONS_FILE = '/mnt/sdb1/projects/dl3/data/melodyst/train/labels.csv'
#SOUND_DIR = '/mnt/sdb1/projects/dl3/data/melodyst/train'
#NOISE_ANNOTATION = '/mnt/sdb1/projects/dl3/data/noise/free-sound/ANNOTATIONS'
#NOISE_DIR = '/mnt/sdb1/projects/dl3/data/noise/free-sound'
#LMDB_AUDIO_PATH = '/mnt/sdb1/projects/dl3/data/melodyst/cache/train_audio_cache.lmdb'
#LMDB_NOISE_PATH = '/mnt/sdb1/projects/dl3/data/cache/noise_cache.lmdb'
#MAP_SIZE = 53687091200

#-- Константы для работы кастомного датасета по 8instruments
PROJECT = 'Medley-solos-DB'
ANNOTATIONS_FILE = '/mnt/sdb1/projects/dl3/data/8instruments/Medley-solos-DB_metadata.csv'
SOUND_DIR = '/mnt/sdb1/projects/dl3/data/8instruments/Medley-solos-DB'
NOISE_ANNOTATION = '/mnt/sdb1/projects/dl3/data/noise/free-sound/ANNOTATIONS'
NOISE_DIR = '/mnt/sdb1/projects/dl3/data/noise/free-sound'
LMDB_AUDIO_PATH='/mnt/sdb1/projects/dl3/data/8instruments/cache/train_audio_cache.lmdb'
LMDB_NOISE_PATH = '/mnt/sdb1/projects/dl3/data/cache/noise_cache.lmdb'
MAP_SIZE = 53687091200
#------------------------------------------
batch_size = 18
learning_rate = 0.000001
scheduler_step_size = 25
scheduler_gamma = 0.5
use_scheduling = False

#for triangle learning scheduling
min_lr =  0.000001
max_lr = 0.00001
step_size_up = 135
w_decay = 1e-7
test_batch_size = 6
saved_epoch = 30
epochs = 15
FIRST_RUN = True
saved_path = "melodyst-w5-emb1024-15.pth"
path = "melodyst-w5-emb1024-15.pth"
knear = 4

inner_dim = 2048
embedding_dim = 1024
sound_len = 20
target_tensor_size = 544

n_mels = 128
n_mfcc = 32
resample_rate = 24000
n_fft = 2048
hop_length = 512

w_factor = 2
L1 = int(8 * w_factor)
L2 = int(32 * w_factor)
L3 = int(128 * w_factor)
L4 = int(512 * w_factor)
L5 = int(1024 * w_factor)

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


def train_epoch(model, device, dataloader, optimizer, loss_fn, miner, k):
    train_loss = 0.0
            
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        
        # Compute loss
        model.zero_grad(set_to_none = True)
        embeddings = model(X)
        pairs = miner(embeddings, y)
        loss = loss_fn(embeddings, y, pairs)
        
        # Backpropagation
        loss.backward()
        #plot_grad_flow(model.named_parameters())
        optimizer.step()
		
        train_loss += loss.item()

    #compute predictions for train data by classes (K-nearest algorithm)
    correct_predictions = 0

    model.eval()
    train_labels = torch.empty((0), dtype = torch.int, device=device)
    train_embeddings = torch.empty((0), dtype = torch.float64, device=device)
    
    #assemble embeddings for train data set
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        
        with torch.no_grad():
             embeddings = model(X)
        
        train_embeddings = torch.cat((train_embeddings, embeddings), dim=0)
        train_labels = torch.cat((train_labels, y), dim=0)

        train_embeddings_cpu = train_embeddings.to(torch.device('cpu'))
        train_labels_cpu = train_labels.to(torch.device('cpu'))

    
    for j in range(train_embeddings_cpu.size()[0]):
        distances = torch.sqrt(torch.sum((train_embeddings_cpu - train_embeddings_cpu[j])**2, dim=1))

        # Устанавливаем расстояние до самого себя как очень большое число чтобы оно не попало в ближайшие
        modified_distances = distances.clone()
        modified_distances[j] = float('inf')
        # Находим k наименьших расстояния
        closest_distances, closest_indices = torch.topk(modified_distances, k, largest=False)

        s_labels = torch.index_select(input=train_labels_cpu, dim=0, index=closest_indices)
        freq = torch.bincount(s_labels)
        pred = torch.argmax(freq)
        if pred.item() == train_labels_cpu[j]: correct_predictions += 1
    return train_loss/len(dataloader.dataset), correct_predictions/train_embeddings.size()[0]


def validate_epoch(model, device, dataloader, loss_fn, miner, k):
    val_loss = 0.0
    val_accuracy = 0.0

    model.eval()
    test_labels = torch.empty((0), dtype = torch.int, device=device)
    test_embeddings = torch.empty((0), dtype = torch.float64, device=device)
    
    #compute loss and assemble embeddings for test data 
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        
        # Compute loss
        with torch.no_grad():
             embeddings = model(X)
        pairs = miner(embeddings, y)
        loss = loss_fn(embeddings, y, pairs)
        val_loss += loss.item()
        # Assemble embeddinds for all test data
        test_embeddings = torch.cat((test_embeddings, embeddings), dim=0)
        test_labels = torch.cat((test_labels, y), dim=0)

    # Compute predictions for test data by classes (K-nearest algorithm)
    correct_predictions = 0

    test_embeddings_cpu = test_embeddings.to(torch.device('cpu'))
    test_labels_cpu = test_labels.to(torch.device('cpu'))
    
    for j in range(test_embeddings_cpu.size()[0]):
        distances = torch.sqrt(torch.sum((test_embeddings_cpu - test_embeddings_cpu[j])**2, dim=1))

        # Устанавливаем расстояние до самого себя как очень большое число чтобы оно не попало в ближайшие
        modified_distances = distances.clone()
        modified_distances[j] = float('inf')
        # Находим k наименьших расстояния
        closest_distances, closest_indices = torch.topk(modified_distances, k, largest=False)

        s_labels = torch.index_select(input=test_labels_cpu, dim=0, index=closest_indices)
        freq = torch.bincount(s_labels)
        pred = torch.argmax(freq)
 
        if pred.item() == test_labels[j].item(): correct_predictions += 1
    
    return val_loss/len(dataloader.dataset), correct_predictions/test_embeddings.size()[0]


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
# Create data loaders

train_dataset = ms.YesDataset(
     dataset_type = 'training',
     annotations_file = ANNOTATIONS_FILE, 
     sound_dir = SOUND_DIR,
     noise_annotation = NOISE_ANNOTATION,
     noise_dir = NOISE_DIR,
     lmdb_audio_path=LMDB_AUDIO_PATH,
     lmdb_noise_path=LMDB_NOISE_PATH,
     map_size = MAP_SIZE,
     sample_rate = resample_rate, 
     sound_len=sound_len, 
     n_mels = n_mels,
	 n_mfcc = n_mfcc,
	 USE_AGUMENTATION = False,
     project = PROJECT)

test_dataset = ms.YesDataset(
     dataset_type = 'test',
     annotations_file = ANNOTATIONS_FILE, 
     sound_dir = SOUND_DIR,
     noise_annotation = NOISE_ANNOTATION,
     noise_dir = NOISE_DIR,
     lmdb_audio_path=LMDB_AUDIO_PATH,
     lmdb_noise_path=LMDB_NOISE_PATH,
     map_size = MAP_SIZE,
     sample_rate = resample_rate, 
     sound_len=sound_len, 
     n_mels = n_mels,
	 n_mfcc = n_mfcc,
	 USE_AGUMENTATION = False,
     project = PROJECT)


m=len(train_dataset)
train_data, val_data = random_split(train_dataset, [int(m-m*0.2)+1, int(m*0.2)])

train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True)
test_dataloader = DataLoader(test_dataset, batch_size = test_batch_size)
val_dataloader = DataLoader(val_data, batch_size = test_batch_size)


train_len = train_dataloader.__len__()
print('train_len:', train_len)

### Set the random seed for reproducible results
torch.manual_seed(0)

torch.set_float32_matmul_precision('high')


# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#init the model
model = mn.wrn_nn(L1=L1, L2=L2, L3=L3, L4=L4, L5=L5, inner_dim=inner_dim, embedding_dim=embedding_dim)
model.to(device)

#model.init_weights()
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
loss_fn = losses.TripletMarginLoss(margin = 0.2, distance = distance, reducer = reducer, embedding_regularizer = regularizer).to(device)
#main_loss_fn = losses.TupletMarginLoss()
#var_loss_fn = losses.IntraPairVarianceLoss()
#loss_fn = losses.MultipleLosses([main_loss_fn, var_loss_fn], weights=[1, 0.5])
miner = miners.TripletMarginMiner(margin = 0.2, distance = distance, type_of_triplets = "semihard")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay)

if use_scheduling:
	#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = min_lr, max_lr = max_lr, step_size_up = step_size_up, cycle_momentum = False, verbose =False)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = scheduler_step_size, gamma = scheduler_gamma)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10,20], gamma = 0.1)

#accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)

if not FIRST_RUN:
	checkpoint = torch.load(saved_path, map_location='cpu')
	model = model.to(device)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	scheduler.load_state_dict(checkpoint['lr'])
	saved_epoch = checkpoint['epochs']
	
else:
	saved_epoch = 0

print('Lets get ready for the rumble with', epochs, end = ' ' )
print('epochs!')

for t in range(saved_epoch, epochs):
    train_loss, mean_accuracy = train_epoch(model=model, device=device, dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn, miner=miner, k=knear)
    val_loss, val_accuracy = validate_epoch(model=model, device=device, dataloader=val_dataloader, loss_fn=loss_fn, miner=miner, k=knear)
    print('\n EPOCH {}/{} \t train loss {:.3f} \t train acc {:.3f} \t val loss {:.3f} \t val acc {:.3f}'.format(t + 1, epochs, train_loss, mean_accuracy, val_loss, val_accuracy))

    #scheduler.step

print("Done!")

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': epochs,
            'lr': scheduler.state_dict()
            }, path)

print(f"Saved PyTorch Model State to  {path}")