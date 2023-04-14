import numpy as np
import torch as th

features = np.loadtxt('clean_features.txt', dtype = 'float32', delimiter = ',')
labels = features = np.loadtxt('clean_labels.txt', dtype = 'float32', delimiter = ',')

class Dataset(th.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx]
    
def build_loader(features, labels):
    dataset = Dataset(features, labels)
    loader = th.utils.data.DataLoader(dataset, batch_size = 66)
    return loader

class NeuralNetwork(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = th.nn.Flatten()
        self.fc1 = th.nn.Linear(54 , 27)
        self.drop = th.nn.Dropout(0.2)
        self.fc2 = th.nn.Linear(27, 1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = th.nn.functional.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

device = "cuda" if th.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)
model.load_state_dict(th.load('model_weights.pth'))
model.eval()
