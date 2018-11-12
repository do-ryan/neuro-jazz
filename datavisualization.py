'''

.midi visualization script using music21

'''

from music21 import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def read_midi(filepath):

    return converter.parse(filepath)

def list_instruments(streamobj):
    instruments = []
    partStream = streamobj.parts.stream()
    # print("List of instruments found on MIDI file:")
    for p in partStream:
        instruments.append(p.partName)
    return instruments

def extract_notes(midi_part):
        # midi_part is a streamiterator object represent a part in the midi and the associated notes with a reference to the original midi stream object

        parent_element = []
        ret = []
        for nt in midi_part.flat.notes: # loop through all notes in stream object
                if isinstance(nt, note.Note): # nt can be either a note or chord object
                        ret.append(max(0.0, nt.pitch.ps)) # ret is note pitch index, constrained to [0, infinity]
                        parent_element.append(nt) # parent_element is note object. note object has volume, pitch, offset, quarterLength (fraction of a quarter note)
                elif isinstance(nt, chord.Chord):
                        for pitch in nt.pitches:
                                ret.append(max(0.0, pitch.ps))
                                parent_element.append(nt)

        return ret, parent_element

def plot_notes_contour(midi):
# midi is a stream object that acts as a container for music elements. a midi file is converted into a stream object in read_midi()

        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 1, 1)

        minPitch = pitch.Pitch('C10').ps # gets pitch index of C10 (highest C)
        maxPitch = 0
        xMax = 0

        # Drawing notes.
        for i in range(len(midi.parts)):
                top = midi.parts[i].flat.notes # top is a music21.stream.iterator object that assists with pulling notes from the ith part of the midi, which is the 'source stream object', accessed by StreamIterator.srcStream. Use methods from http://web.mit.edu/music21/doc/moduleReference/moduleStreamIterator.html to pull data from the source stream object.
                y, parent_element = extract_notes(top) # y is note pitch index, parent_element is note/chord object
                if (len(y) < 1): continue

                x = [n.offset for n in parent_element] # offset is time offset from t=0 of note

                ax.scatter(x, y, alpha=0.6, s=7)

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
        plt.title('Voices motion approximation, each color is a different instrument, red lines show each octave')
        plt.show()


def main():

        example1 = read_midi("training_data/authentic/20.midi")
        #example1.plot('horizontalbar')
        print_parts_countour(example1)
        print(list_instruments(example1))

        return

if __name__ == "__main__":
        main()