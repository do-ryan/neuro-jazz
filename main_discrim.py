# Main file

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from models import CNN, GAN
from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training loop
def load_data(data_train, labels_train, data_val, labels_val, batchsize):
    train_set = Dataset(data_train, labels_train)
    valid_set = Dataset(data_val, labels_val)
    train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batchsize, shuffle=False)
    return train_loader, valid_loader

def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set."""

    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0

    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs = inputs.permute(0, 2, 1)
        labels = np.asarray(labels)
        labels = torch.from_numpy(labels)

        outputs = net(inputs)
        loss = criterion(input=outputs, target=labels)
        outputs = outputs.detach().numpy()
        predictions = outputs.argmax(axis=1)
        corr = predictions != labels

        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)

    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)

    return err, loss


def main():
    MaxEpochs = 15
    lr = 0.5 
    batch_size = 1

    net = CNN().to(device)
    net = net.cuda()

    loss_fnc = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    data = np.load('./data/instance_test.npy')
    labels = np.load('./data/labels_test.npy')

    np.random.seed(0)
    torch.manual_seed(0)
    data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=0.2, random_state=0)  
    train_loader, val_loader = load_data(data_train, labels_train, data_val, labels_val, batch_size)

    train_err = np.zeros(MaxEpochs)
    train_loss = np.zeros(MaxEpochs)
    epoch_arr = []

    # Training loop
    for epoch in range(MaxEpochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        epoch_arr.append(epoch)

        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.permute(0, 2, 1)
            labels = np.asarray(labels)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            labels = torch.from_numpy(labels)
	    print(outputs)
	    print(labels)
            
            loss = loss_fnc(input=outputs, target=labels.double().cuda())
            loss.backward()
            optimizer.step()
            
            outputs = outputs.detach().cpu().numpy() # output of the model

            # Calculate the statistics
            corr = (outputs > 0.5).squeeze().astype(int) != labels

            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)
        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i + 1)
        print("Epoch {} | Train acc: {} | Train loss: {}".format(epoch + 1, 1 - train_err[epoch], train_loss[epoch]))


if __name__ == "__main__":
    main()
