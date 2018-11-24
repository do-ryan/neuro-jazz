# Main file

from datavisualization import list_instruments
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from models import CNN, GAN

# Training loop
def load_data(data, labels, batchsize):
    train_set = Dataset(data, labels)
    train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
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

def generate(input, output, epochs):
    GAN(input, 20, output)


def main():
    MaxEpochs = 20
    lr = 0.001
    batch_size = 10
    latent_size = 64
    hidden_size = 256
    output_size = 46842

    d_net = CNN()
    g_net = GAN(latent_size, hidden_size, output_size)
    loss_fnc = nn.BCEWithLogitsLoss()
    d_optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    data = np.load('./data/instances.npy')
    labels = np.load('./data/labels.npy')

    train_loader = load_data(data, labels, batch_size)

    # Training loop
    for epoch in range(MaxEpochs):

        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            inputs = inputs.permute(0, 2, 1)
            real_labels = torch.ones(batch_size, 1)             # all training data for real images set to 1
            fake_labels = torch.zeros(batch_size, 1)            # all output from generator fake so 0

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            ## Train Discriminator
            real_output = d_net(inputs)
            d_loss_real = loss_fnc(real_output, real_labels)

            z = torch.randn(batch_size, latent_size)
            fake_output = g_net(z)
            predictions = d_net(fake_output)
            d_loss_fake = loss_fnc(predictions, fake_labels)

            d_loss_tot = d_loss_real + d_loss_fake
            d_loss_tot.backwards()
            d_optimizer.step()

            ## Train Generator
            z = torch.randn(batch_size, latent_size)
            fake_output = g_net(z)
            predictions = d_net(fake_output)
            g_loss = loss_fnc(predictions, real_labels)         # bc we want to train G to minimize 1-D(G(z)) so we will maximize D(G(z))
            g_loss.backwards()
            g_optimizer.step()

            # Calculate the statistics

            print("Epoch {} | Step {} | d_loss: {} | g_loss: {}".format(epoch + 1, i+1, d_loss_tot.item(), g_loss.item()))


if __name__ == "__main__":
    main()
