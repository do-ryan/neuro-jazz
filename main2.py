import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from models import CNN, GAN
device = torch . device( "cuda:0" if torch . cuda . is_available() else "cpu" )
from music21 import *

def main():
    batch_size = 1
    latent_size = 16
    output_size = 64*24
    z = torch.randn(batch_size, latent_size)
    model = GAN(latent_size, 10, output_size)

    output = model(z).detach().numpy()
    # generated numpy array representing a sample of music

    numpy_to_midi(output)


def numpy_to_midi(numpy_input):
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



    s1.write("midi", "gen_data/file1.midi")

if __name__ == "__main__":
    main()