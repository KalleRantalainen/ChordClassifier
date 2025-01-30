# Create testing, training and validation sets from the wav files.
# So split each song into small windows and assign chord value for that window.
# Calculate the Constant Q spectrogram of that winodw. This spectrogram is
# the feature, the chord is the label.

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from file_io import get_all_file_names, read_wav_file

def calculate_const_Q(y, sr):
    # TODO: Laske constant Q spectrogrammi
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, bins_per_octave=24, n_bins=84))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    return cqt_db

def calculate_mel_spectro():
    # TODO: laske mel spectrogrammi, tarvitaanko?
    pass

def process_data():
    # TODO: splittaa lyhyihin ikkunoihin, laske spectrogrammi
    # Asenna label tälle ikkunnalle.
    train_file_names = get_all_file_names('train')
    for idx, train_file in enumerate(train_file_names):
        sr, y = read_wav_file(train_file)
        constq = calculate_const_Q(y, sr)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(constq, sr=sr, x_axis="time", y_axis="cqt_note")
        plt.colorbar(label="dB")
        plt.title("Constant-Q Transform (CQT) Spectrogram")
        plt.show()
        break

process_data()