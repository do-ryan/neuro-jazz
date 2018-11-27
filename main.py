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

def load_d_model(lr, batch_size):
    model = CNN(batch_size = batch_size).to(device)
    loss_fnc = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer

def load_g_model(input_size, generated_size, lr, batch_size, hidden_size, pitch_range):
    model = GAN(input_size=input_size, hidden_size = hidden_size, output_size=generated_size, batch_size=batch_size, pitch_range=pitch_range).to(device)
    loss_fnc = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8) # batch size (for both generator and discriminator)
    parser.add_argument('--lr_d', type=float, default=0.4) # discriminator learning rate
    parser.add_argument('--lr_g', type=float, default=0.75) # generator learning rate
    parser.add_argument('--epochs', type=int, default=25)  # change default value to change hyperparameter value, or in run/debug configuration
    # or in terminal "python main.py --lr 0.001". "python main.py --help". have to use argparse
    parser.add_argument('--latent_size', type=int, default=8) # size of random vector input to generator
    parser.add_argument('--hidden_size', type=int, default=1000) # number of neurons in hidden layers of generator MLP
    args = parser.parse_args()

    #test_generator_to_midi(1, 16, 64 * 24)
    data = np.load('./data/instance_authentic.npy')
    labels = np.load('./data/labels_authentic.npy') #don't need labels, all of the training data is authentic

    test_generator_to_midi(batch_size=1, latent_size=args.latent_size, output_size=data.shape[2], pitch_range=data.shape[1], hidden_size=args.hidden_size)

    train_loader = load_data(data, labels, args.batch_size)

    model_d, loss_fnc_d, optimizer_d = load_d_model(lr=args.lr_d, batch_size= args.batch_size)
    model_g, loss_fnc_g, optimizer_g = load_g_model(pitch_range=data.shape[1], input_size=args.latent_size, generated_size=data.shape[2], lr=args.lr_g, batch_size = args.batch_size, hidden_size=args.hidden_size)

    for epoch in range(args.epochs):
        tot_corr = 0
        for i, batch in enumerate(train_loader, 0):
            inputs, labels = batch
            inputs = inputs.to(device)

            authentic_labels = torch.ones(args.batch_size)  # labels for authentic data real so automatically set to 1
            fake_labels = torch.zeros(args.batch_size)  # labels for generated data fake so automatically set to 0

            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            # initialize optimizers

            z = torch.randn(args.batch_size, args.latent_size)
            # create latent vectors
            fake_output = model_g(z.float())
            # generate data

            # MODEL FWD AND LOSS CALCULATIONS
            #################################
            authentic_predictions = model_d(inputs)
            d_loss_auth = loss_fnc_d(authentic_predictions, authentic_labels.double())
            # calculate loss of classifying authentic data

            fake_predictions = model_d(fake_output.double())
            d_loss_fake = loss_fnc_d(fake_predictions, fake_labels.double())
            # calculate loss of classifying generated data

            authentic_predictions = authentic_predictions.detach().cpu().numpy()
            fake_predictions = fake_predictions.detach().cpu().numpy()

            corr = int(((fake_predictions > 0.5).squeeze().astype(int) == fake_labels).sum()) + int(((authentic_predictions > 0.5).squeeze().astype(int) == authentic_labels).sum())
            # sum up correct predictions
            tot_corr += corr
            # correct discriminator predictions

            # TRAIN DISCRIMINATOR
            #####################
            d_loss_tot = d_loss_fake + d_loss_auth # sum prediction loss of both sub data sets
            d_loss_tot.backward()
            optimizer_d.step() # improve discriminator

            # TRAIN GENERATOR
            #################
            z = torch.randn(args.batch_size, args.latent_size)
            fake_output = model_g(z.float())
            new_auth_predictions = model_d(fake_output.double())
            g_loss = loss_fnc_g(new_auth_predictions, authentic_labels.double()) # minimizes loss of predicting fake as real so the generator is tricking the discriminator
            g_loss.backward() # generator result prediction loss has a reference to generator model (function)
            # the gradient is computed wrt generator weights https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
            optimizer_g.step() # improve the generator.
            print("D loss: ", d_loss_tot.data[0].item(), " G loss: ", g_loss.data[0].item()) # print losses for current batch

        print("Epoch ", epoch+1, "Train acc:{}".format(float(tot_corr) / (len(train_loader.dataset)*2)))

    for i, sample in enumerate(fake_output, 0):
        #fake_output is a collection of samples of a batch size
        numpy_to_midi(sample.detach().cpu().numpy(), "gen_data/sample{}.midi".format(i))



def test_generator_to_midi(batch_size, latent_size, output_size, pitch_range, hidden_size):
    # output size is length sample in terms of subdivisions
    z = torch.randn(batch_size, latent_size)
    # create random vector

    model = GAN(input_size=latent_size, hidden_size=hidden_size, output_size=output_size, batch_size = batch_size, pitch_range=pitch_range)

    output = model(z).detach().numpy()[0]
    # generated numpy array representing a sample of music

    numpy_to_midi(output, "gen_data/test1.midi")

def numpy_to_midi(numpy_input, filewrite_path):
    # writes resultant midi to filewrite_path

    #initialization
    s1 = stream.Stream()
    SUBDIVISION = 24
    VOLUME_SCALING = 20
    bRest = True

    #print(numpy_input) display np array before converting to stream -> midi

    for i in range(numpy_input.shape[0]):
        currentNoteLength = 0 # in subdivisions
        for j in range(numpy_input.shape[1]):
            if numpy_input[i][j] != 0:
                if bRest == True: # when state changes from rest to note
                    n = note.Note() # instantiate note object
                    p = pitch.Pitch() # instantiate pitch object
                    p.ps = i+48 # using pitch space representation
                    n.pitch = p # assign pitch in pitch space representation to note pitch
                    n.volume.velocityScalar = numpy_input[i][j]*VOLUME_SCALING
                    n.offset = j/SUBDIVISION
                    currentNoteLength += 1
                else: # if already in current note
                    currentNoteLength += 1
                bRest = False # current element is a note so bRest is false
            else:
                if bRest == False: # when state changes from note to rest:
                    d = duration.Duration(currentNoteLength/SUBDIVISION)
                    n.duration = d
                    if n.duration.quarterLength >= 0.25: # filter notes shorter than a sixteenth
                        s1.insert(n.offset, n) # insert note at its offset to stream, and will combine with a note if there is already one at that position
                bRest = True # bRest is true if a note doesn't take place on the current element in the array
                currentNoteLength = 0

        print("parsing numpy row ", i)

    s1.write("midi", filewrite_path)
    print ("New midi created at: ", filewrite_path)

if __name__ == "__main__":
    main()
