import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import deeptime as dt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from deeptime.util.torch import MLP

data = np.load('../data.npy')
di1 = 10
di2 = 18
di_slow_data = np.stack([np.concatenate(data)[:,di1],np.concatenate(data)[:,di2]], axis =1)
data_list = [data[i] for i in range(data.shape[0])]
print('Data loaded')

'''
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
torch.set_num_threads(8)
print(f"Using device {device}")
'''

tau = 1
batch_size = 6
val_ratio = 0.15
learning_rate = 1e-5
nb_epoch = 10

from deeptime.util.data import TrajectoryDataset, TrajectoriesDataset
dataset = TrajectoriesDataset.from_numpy(tau, data_list)

n_val = int(len(dataset)*val_ratio)
train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset)-n_val, n_val])

lobe = nn.Sequential(
    nn.BatchNorm1d(data[0].shape[1]),
    nn.Linear(data[0].shape[1], 80), nn.ELU(),
    nn.Linear(80, 60), nn.ELU(),
    nn.Linear(60, 40), nn.ELU(),
    nn.Linear(40, 20), nn.ELU(),
    nn.Linear(20, 20), nn.ELU(),
    nn.Linear(20, 6),
    nn.Softmax(dim=1)  # obtain fuzzy probability distribution over output states
)

#lobe = lobe.to(device=device)

vampnet = dt.decomposition.deep.VAMPNet(lobe=lobe, learning_rate=learning_rate)#, device=device)

loader_train = DataLoader(train_data, batch_size = batch_size, shuffle=True)
loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
print('Ready to train')
model = vampnet.fit(data_loader=loader_train, n_epochs=nb_epoch, validation_loader=loader_val, progress=tqdm).fetch_model()

import pickle
with open('vnet_uncat.pkl', 'wb') as file:
	pickle.dump(model, file)

plt.loglog(*vampnet.train_scores.T, label='Training')
plt.loglog(*vampnet.validation_scores.T, label='validation')
plt.xlabel('Step')
plt.ylabel('Score')
plt.legend();
plt.savefig('vnet_uncat_loss.jpg')

state_probabilities = model.transform(data)

'''
f, axes = plt.subplots(1, x, figsize=(8, 10))
for i, ax in enumerate(axes.flatten()):
    ax.scatter(*di_slow_data.T, c=state_probabilities[..., i])
    ax.set_title(f'State {i+1}')
plt.savefig('vnet_uncat_states.jpg')

for ix, (mini, maxi) in enumerate(zip(np.min(state_probabilities, axis=0), 
                                      np.max(state_probabilities, axis=0))):
    print(f"State {ix+1}: [{mini}, {maxi}]")
'''

assignments = state_probabilities.argmax(2)
plt.scatter(*di_slow_data.T, c=assignments, s=.5, cmap='jet')
plt.title('Transformed state assignments');
plt.savefig('vnet_uncat_assignments.jpg')

