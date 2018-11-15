'''

.midi visualization script using music21

'''

from music21 import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy as np

def read_midi(filepath):

    return converter.parse(filepath) # returns a stream object

def list_instruments(streamobj):
    instruments = []
    partStream = streamobj.parts.stream() #partstream is a stream object representing part
    # print("List of instruments found on MIDI file:")
    for p in partStream:
        instruments.append(p.partName)
    return instruments

def extract_notes(midi_part):
        # midi_part is a streamiterator object represent a part in the midi and the associated notes with a reference to the original midi stream object
        # returns all pitches and note objects in a part of the midi

        parent_element = []
        ret = []
        for nt in midi_part.flat.notes: # loop through all notes in stream iterator object
                if isinstance(nt, note.Note): # nt can be either a note or chord object
                        ret.append(min(max(0.0, nt.pitch.ps), pitch.Pitch('C10').ps)) # ret is note pitch index, constrained to [0, 132 (C10)]
                        parent_element.append(nt) # parent_element is note object. note object has volume, pitch, offset, quarterLength (fraction of a quarter note)
                elif isinstance(nt, chord.Chord):
                        for pitch_a in nt.pitches:
                                ret.append(max(0.0, pitch_a.ps))
                                parent_element.append(nt)

        return ret, parent_element

def plot_notes_contour(midipart, instrument, midi_path, time_range_to_plot):
        # plots notes for a single part of midi.parts on pitch vs. time axes with colour representing note volume
        # midipart is a stream object that acts as a container for music elements. a midi file is converted into a stream object in read_midi()
        # instrument is a string representing instrument
        # midi_path is name of file
        # time_range_to_plot is a 2-element list representing min and max fraction of total time to plot

        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 1, 1)

        minPitch = pitch.Pitch('C10').ps # gets pitch index of C10 (highest C)
        maxPitch = 0
        xMax = 0


        # Drawing notes.
        top = midipart.flat.notes # top is a music21.stream.iterator object that assists with pulling notes from the ith part of the midi, which is the 'source stream object', accessed by StreamIterator.srcStream. Use methods from http://web.mit.edu/music21/doc/moduleReference/moduleStreamIterator.html to pull data from the source stream object.
        y, parent_element = extract_notes(top) # y is note pitch index, parent_element is note/chord objects

        x = [n.offset for n in parent_element] # offset is time offset from t=0 of note
        volume = [n.volume.velocityScalar for n in parent_element] # volume is a float

        cm = plt.cm.get_cmap('RdYlBu') # add colormap
        # plot = ax.scatter(x[int((len(x)*time_range_to_plot[0])):int(len(x)*time_range_to_plot[1])], y[int(len(y)*time_range_to_plot[0]):int(len(y)*time_range_to_plot[1])], alpha=0.6, s=7, c=volume, cmap=cm)
        #above: plot fraction of points

        plot = ax.scatter(x, y, alpha=0.6, s=7, c=volume, cmap=cm)

        aux = min(y)
        if (aux < minPitch):
                minPitch = aux
        aux = max(y)
        if (aux > maxPitch):
                maxPitch = aux
        aux = max(x)
        if (aux > xMax):
                xMax = aux
        # set min/max pitches for octave line plotting (constrained to [0, C10])


        for i in range(1, 10):
                linePitch = pitch.Pitch('C{0}'.format(i)).ps
                if (linePitch > minPitch and linePitch < maxPitch):
                        ax.add_line(mlines.Line2D([0, xMax], [linePitch, linePitch], color='red', alpha=0.1))
        # plot octave lines between minPitch and maxPitch

        plt.ylabel("Note index (each octave has 12 notes)")
        plt.xlabel("Number of beats (quarter notes)")
        plt.title('{} part in {}. \nNote motion approximation, note volume represented by colour, red lines show each C octave'.format(instrument, midi_path))
        plt.colorbar(plot)
        plt.show()

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
def main():

        midi_path = "training_data/authentic/5.midi"

=======
>>>>>>> 7a8353031e7d7dfbd428aa6c25dda20bd8847200
=======
>>>>>>> 7a8353031e7d7dfbd428aa6c25dda20bd8847200
=======
>>>>>>> 7a8353031e7d7dfbd428aa6c25dda20bd8847200
=======
>>>>>>> 7a8353031e7d7dfbd428aa6c25dda20bd8847200
# def grid_plot()
        # take in a volume(pitch, time) numpy array and plots the data in a grid format

def main():

        midi_path = "training_data/authentic/10.midi"
        example1 = read_midi(midi_path)
        # parse midi file

        for i, part in enumerate(example1.parts):
                plot_notes_contour(part, instrument=list_instruments(example1)[i], midi_path=midi_path, time_range_to_plot=[0, 0.2]) # plot all parts in example 1
        # plot notes/volumes of each part in midi

        p = graph.plot.HistogramQuarterLength(example1, tickFontSize=4, title='Note duration (in quarter notes) histogram for \n{}'.format(midi_path))
        p.run()
        # plot histogram of note durations

        preprocessed_data = np.load('./data/instances.npy')

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 7a8353031e7d7dfbd428aa6c25dda20bd8847200
=======

>>>>>>> 7a8353031e7d7dfbd428aa6c25dda20bd8847200
=======

>>>>>>> 7a8353031e7d7dfbd428aa6c25dda20bd8847200
=======

>>>>>>> 7a8353031e7d7dfbd428aa6c25dda20bd8847200
        print(list_instruments(example1))

        return

if __name__ == "__main__":
        main()