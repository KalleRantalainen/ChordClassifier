# Create testing, training and validation sets from the wav files.
# So split each song into small windows and assign chord value for that window.
# Calculate the Constant Q spectrogram of that winodw. This spectrogram is
# the feature, the chord is the label.

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from file_io import get_all_file_names, read_wav_file, get_chords_and_times

def calculate_const_Q(y, sr):
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=256, bins_per_octave=24, n_bins=84))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    return cqt_db

def one_hot_encode_target(chord, all_chords):
    # Find the index of the chord in the all_chords list
    chord_index = all_chords.index(chord)
    # Vector of zeros
    one_hot_vector = np.zeros(len(all_chords), dtype=int)
    # Set the chord's value to 1
    one_hot_vector[chord_index] = 1
    return one_hot_vector

def calculate_mel_spectro():
    # TODO: laske mel spectrogrammi, tarvitaanko?
    pass

def calculate_features(qspec, y, sr, file_name):
    # Number of bins for each feature. 6 bins corresponds to approx 100ms
    number_of_bins = 6
    # Total number of bins in the spectrogram
    qspec_bins = qspec.shape[1]
    # Calculate how many feature/target pairs can be formed from this song.
    num_of_features = qspec_bins // number_of_bins
    # file length in seconds
    y_len_seconds = len(y) / sr
    # How long each bin is in seconds
    bin_len_seconds = y_len_seconds / qspec_bins

    # Convert the name from "../data/test/name.wav" to "test/name.wav"
    json_format_file_name = file_name.replace("../data/", "")
    # Get the chords and their timestamps from the file
    chords, chord_times = get_chords_and_times(json_format_file_name)
    # Add start and end to the chord times
    chord_times.insert(0, 0)
    chord_times.append(y_len_seconds)
    # Add None to the start and end of the chords
    chords(0, None)
    chords.append(None)

    # If the song has signle E chord, the format is now for example
    # chords = [None, E, None]
    # chord_times = [0, 0.5, 10]
    # So the is nothing from 0 to 0.5 seconds, then there is E from 0.5 to 10 seconds
    # and nothing after that since the song has ended

    if chords == None or chord_times == None:
        return

    # Loop through the bins and store feature/target pairs
    for i in range(num_of_features - 1):
        # Start and end idx for the current bins
        start_idx = i * number_of_bins
        end_idx = (i + 1) * number_of_bins
        # Start and end times for the current bins
        start_sec = i * bin_len_seconds
        end_sec = i * bin_len_seconds
        # Exract current bins from the spectrogram
        current_bins = qspec[start_idx:end_idx]

        # Placeholder for the active chord
        active_chord = None
        for j in range(len(chord_times) - 1):
            # Check if the current window has only one chord
            if chord_times[j] <= start_sec and chord_times[j + 1] >= end_sec:
                # Single chord active in this window
                active_chord = chords[j]
            # There are two chords present during this window
            else:
                # Overlap starts when the current chord ends
                overlap_start = chord_times[j + 1]
                # Overlap ends when the current window ends
                overlap_end = end_sec
                # Calculate the overlap duration
                overlap_duration = overlap_end - overlap_start
                # If overlap takes more than half of the window's duration, then
                # the overlapping chord is the active one.
                if overlap_duration > (end_sec - start_sec) / 2:
                    active_chord = chords[j + 1]
                else:
                    # If the overlap is less than half of the window duration, the active chord
                    # is the current chord
                    active_chord = chords[j]
            # Active chord found, break the loop
            break

        # Handle case where no chord is active
        if active_chord is None:
            # Do not write any features
            pass
        else:
            # Active chord was found -> create feature/target pair file.
            pass




def process_data(split):
    """
    Calculate the feaures from the data.
    1. Calculate the a spectrogram for a whole song.
    2. Split the spectrogram into short windows
    3. Repeat for every file
    """
    file_names = get_all_file_names(split)
    for idx, file_name in enumerate(file_names):
        sr, y = read_wav_file(file_name)
        if sr != 16000:
            continue
        constq = calculate_const_Q(y, sr)
        bins = constq.shape[1]
        samples_per_bin = len(y) / bins
        print(file_name)
        print("sr:", sr)
        print("constQ shape:", constq.shape)
        print("Length of one bin:", samples_per_bin/sr*6, "s")
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(constq, sr=sr, x_axis="time", y_axis="cqt_note")
        plt.colorbar(label="dB")
        plt.title("Constant-Q Transform (CQT) Spectrogram")
        plt.show()
        break

# For testing
def main():

    process_data("test")

if __name__ == "__main__":
    main()