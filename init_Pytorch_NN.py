import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(FeedForwardNN, self).__init__()
        self.layers = nn.ModuleList()
        
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.ReLU())
            prev_size = hidden_size

        self.layers.append(nn.Linear(prev_size, num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 
    
    # def train(self, num_epochs, batch_size, learning_rate, training_data):
    def train(self, num_epochs, input_size, learning_rate, train_dataloader, debug=False, info_skip=100):
        epoch_losses = []
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in tqdm(range(num_epochs)):
            batch_losses = []
            for i, (images, labels) in tqdm(enumerate(train_dataloader)):
            # for i, (images, labels) in tqdm(enumerate(training_data)):
                # Flatten the images and convert to PyTorch tensors
                images = images.view(-1, input_size)
                
                # Forward pass
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                batch_losses.append(loss.item())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if debug & ((i+1) % info_skip == 0):
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
                    # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(training_data)}], Loss: {loss.item():.4f}')
            print("batch loss: ", batch_losses)
            epoch_losses.append(np.mean(batch_losses))
            print("epoch loss: ", epoch_losses[-1])

        print("Training complete.")
        return epoch_losses

    def test(self, test_loader, input_size):
        # test_loss = 0
        correct = 0
        cross_ent_errors = []
        classification_matches = []
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, input_size)
                output = self.forward(data)
                # print("output: ", output)
                # print("target: ", target)
                # test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                loss = criterion(output, target)
                cross_ent_errors.append(loss)  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                # print("pred: ", pred.eq(target.view_as(pred)).item())
                # classification_matches.append(pred.eq(target.view_as(pred)))
                classification_matches = classification_matches + pred.eq(target.view_as(pred)).tolist()

        # test_loss /= len(test_loader.dataset)
        # accuracy = 100. * correct / len(test_loader.dataset)

        # print('Average cross entropy loss:', np.mean(cross_ent_errors))
        # print(f"Classification accuracy: {(sum(classification_matches)/len(test_loader))*100}%")

        return cross_ent_errors, classification_matches

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))