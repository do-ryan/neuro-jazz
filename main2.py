import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from models import CNN, GAN
device = torch . device( "cuda:0" if torch . cuda . is_available() else "cpu" )
from music21 import *

def load_data(data, labels, batchsize):
    train_set = Dataset(data, labels)
    train_loader = DataLoader(train_set, batch_size=batchsize)
    return train_loader

def load_d_model(lr):
    model = CNN()
    loss_fnc = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer

def load_g_model(input_size, generated_size, lr):
    model = GAN(input_size=input_size, hidden_size = 10, output_size=24)
    loss_fnc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_d', type=float, default=0.01)
    parser.add_argument('--lr_g', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)  # change default value to change hyperparameter value, or in run/debug configuration
    # or in terminal "python main.py --lr 0.001". "python main.py --help". have to use argparse
    parser.add_argument('--latent_size', type=int, default=8)
    parser.add_argument('--generated_size', type=int, default=24*4*8)
    args = parser.parse_args()

    #test_generator_to_midi(1, 16, 64 * 24)
    data = np.load('./data/instances.npy')
    labels = np.load('./data/labels.npy') #don't need labels, all of the training data is authentic

    train_loader = load_data(data, labels, args.batch_size)

    model_d, loss_fnc_d, optimizer_d = load_d_model(lr=args.lr_d)
    model_g, loss_fnc_g, optimizer_g = load_g_model(input_size=args.latent_size, generated_size=args.generated_size, lr=args.lr_g)

    t = 0

    for epoch in range(args.epochs):
        tot_corr = 0
        accum_loss = 0
        for i, batch in enumerate(train_loader, 0):
            inputs, labels = data

            authentic_labels = torch.ones(batch_size, 1)  # all training data for real data set to 1
            fake_labels = torch.zeros(batch_size, 1)  # all output from generator fake so 0

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            # initialize optimizers

            z = torch.randn(args.batch_size, args.latent_size)
            # create latent vectors
            fake_output = model_g(z)
            # generate data

            # MODEL FWD AND LOSS CALCULATIONS
            #################################
            authentic_predictions = model_d(inputs)
            d_loss_auth = loss_fnc_d(authentic_predictions, authentic_labels)
            # calculate loss of classifying authentic data

            fake_predictions = model_d(fake_output)
            d_loss_fake = loss_fnc_d(fake_predictions, fake_labels)
            # calculate loss of classifying generated data

            corr = (fake_predictions > 0.5).squeeze().long() == fake_labels + (authentic_predictions > 0.5).squeeze().long() == authentic_labels
            tot_corr += corr
            # correct discriminator predictions

            # TRAIN DISCRIMINATOR
            #####################
            d_loss_tot = d_loss_real + d_loss_auth # sum prediction loss of both sub data sets
            d_loss_tot.backwards()
            d_optimizer.step()

            # TRAIN GENERATOR
            #################
            z = torch.randn(batch_size, latent_size)
            fake_output = model_g(z)
            new_auth_predictions = model_d(fake_output)
            g_loss = loss_fnc_g(new_auth_predictions, real_labels) # minimizes loss of predicting fake as real so the generator is tricking the discriminator
            g_loss.backwards() # generator result prediction loss has a reference to generator model (function)
            # so the gradient is computed wrt generator weights https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

        print("Train acc:{}".format(float(tot_corr) / (len(train_loader.dataset)*2)))

    for i, sample in enumerate(fake_output, 0):
        #fake_output is a collection of samples of a batch size
        numpy_to_midi(sample, "gen_data/sample{}.midi".format(i))



def test_generator_to_midi(batch_size, latent_size, output_size):
    z = torch.randn(batch_size, latent_size)
    # create random vector

    model = GAN(input_size=8, hidden_size=10, output_size=24*4*8)

    output = model(z).detach().numpy()
    # generated numpy array representing a sample of music

    numpy_to_midi(output, "gen_data/sample1.midi")


def numpy_to_midi(numpy_input, filewrite_path):
    # returns stream
    s1 = stream.Stream()
    SUBDIVISION = 24

    for i in range(numpy_input.shape[0]):
        for j in range(numpy_input.shape[1]):
            if numpy_input[i][j] != 0:
                # placeholder for note durations
                n = note.Note()
                p = pitch.Pitch()
                p.ps = i # using pitch space representation
                n.pitch = p
                n.volume.velocityScalar = numpy_input[i][j]*50
                n.offset = j/SUBDIVISION
                s1.append(n)
        print(i)

    s1.write("midi", filewrite_path)

if __name__ == "__main__":
    main()