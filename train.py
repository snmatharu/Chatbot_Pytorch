import json

from torch.cuda import is_available
from bot import tokenize, stem, bag_of_words
from model import NeuralNetwork
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataset

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', ',', '.']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words)) # set removes duplicate
tags = sorted(tags)

X_train = []
y_train = []

for (pattern_sentance, tag) in xy:
    bag = bag_of_words(pattern_sentance, all_words)
    X_train.append(bag)
    label = tags.index(tag) #it will give label
    y_train.append(label) #crossEntropyloss

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples

batch_size = 8
hidden_size = 5
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001

num_epochs = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        
        words = words.to(device)
        labels = labels.to(device)

        # forward

        outputs = model(words)
        loss = criterion(outputs, labels)

        # backwrds and optimizer steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(epoch)
    if((epoch +1)% 100 == 0):
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
        

print(f'Final loss: {loss.item():.4f}')

data = {
    "model_state" : model.state_dict(),
    "input_size" : input_size,
    "output_size" : output_size,
    "hidden_size" : hidden_size,
    "all_words" : all_words,
    "tags" : tags
}

File ="data.pth"
torch.save(data, File)

print(f'training complete. File saved to {File}')