# Main file

from datavisualization import list_instruments
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from models import CNN

def main:


if __name__ == "__main__":
    main()

# Training loop
def load_data(data, labels, batchsize):
    train_set = Dataset(data, labels)
    train_loader = DataLoader(train_set, batch_size=batchsize)
    return train_loader

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
    MaxEpochs = 20
    lr = 0.1
    batch_size = 64

    net = CNN()
    loss_fnc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    data = np.load('./data/instances.npy')
    labels = np.load('./data/labels.npy')
    train_loader = load_data(data, labels, batch_size)

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
            inputs = inputs.permute(0, 2, 1)
            labels = np.asarray(labels)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            labels = torch.from_numpy(labels)

            loss = loss_fnc(input=outputs, target=labels)
            loss.backward()
            optimizer.step()

            outputs = outputs.detach().numpy() # output of the model

            # Calculate the statistics
            corr = (outputs > 0.0).squeeze().astype(int) != labels

            outputs = outputs.detach().numpy()
            predictions = outputs.argmax(axis=1)

            # Calculate the statistics
            corr = predictions != labels

            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i + 1)

        print("Epoch {} | Train acc: {}".format(epoch + 1, 1 - train_err[epoch]))


if __name__ == "__main__":
    main()
