import numpy as np
import torch as th
import matplotlib.pyplot as plt
import math

features = np.loadtxt('clean_features.txt', dtype = 'float32', delimiter = ',')
labels = np.loadtxt('clean_labels.txt', dtype = 'float32', delimiter = ',')

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

model = NeuralNetwork().to("cpu")
model.load_state_dict(th.load('model_weights.pth'))
model.eval()

loader = build_loader(features, labels)

fig, ax = plt.subplots()
chart_x = np.arange(10)
chart_y = np.zeros(10, dtype = int)

size = len(loader.dataset)
numBatches = len(loader)
model.eval()
correct = 0
with th.no_grad():
    for X, y in loader:
        X, y = X.to("cpu"), y.to("cpu")
        prediction = model(X)
        prediction = prediction.squeeze()
        prediction = th.sigmoid(prediction)
        for i in range(np.shape(prediction)[0]):
            num = prediction[i].item() * 10
            num = math.floor(num)
            chart_y[num] += 1
        prediction = (prediction > 0.5).type(th.float)
        correct += (prediction == y).type(th.float).sum().item()
correct /= size
print(f"Model Accuracy: {(100 * correct):>0.1f}%")

#Distribution of model prediction values
ax.bar(chart_x, chart_y, width = 1, edgecolor = "white", linewidth = 0.7)
ax.set(xlim = (-0.5, 9.5), xticks = np.arange(0, 10))
plt.show(block = True)
