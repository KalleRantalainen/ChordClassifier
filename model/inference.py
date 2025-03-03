# File to test the model using low quality electric quitar recordings.

import numpy as np
import librosa
import torch
import os
import torch.nn.functional as F
import sounddevice as sd

from dataset import ChordDataset
from cnn import ChordCNN
from file_io import read_wav_file, get_common_chords
from process_files import calculate_const_Q

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    # Change the sampling rate to 16000 as the training files have that
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y_resampled, 16000

def play_audio(y, sr):
    print("Playing audio...")
    sd.play(y, samplerate=sr)
    sd.wait()

def extract_features(y, sr):
    qspec = calculate_const_Q(y, sr)
    return qspec

def create_feature_windows(qspec, signal_length_sec, number_of_bins=6):
    qspec_bins = qspec.shape[1]
    num_of_features = qspec_bins // number_of_bins
    bin_length_sec = signal_length_sec / qspec_bins

    features = []
    for i in range(num_of_features - 1):
        start_idx = i * number_of_bins
        end_idx = (i + 1) * number_of_bins
        start_sec = start_idx * bin_length_sec
        end_sec = end_idx * bin_length_sec
        features_and_times = (qspec[:, start_idx:end_idx], start_sec, end_sec)
        features.append(features_and_times)

    return features

def predict_chords(model, features, common_chords, threshold = 0.5):
    predictions = []
    with torch.no_grad():
        for feature_tuple in features:
            feature = feature_tuple[0]
            start_time = feature_tuple[1]
            end_time = feature_tuple[2]
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            output = model(feature_tensor)
            # Apply the softmax manually since it is not applied in the 
            # forward method.
            probabilities = F.softmax(output, dim=1)
            #print("Output:", probabilities)
            predicted_index = torch.argmax(probabilities, dim=1).item()
            predicted_probability = probabilities[0, predicted_index].item()
            if predicted_probability < threshold:
                pred_tuple = (None, start_time, end_time)
                predictions.append(pred_tuple)
            else:
                predicted_chord = common_chords[predicted_index]
                pred_tuple = (predicted_chord, start_time, end_time)
                predictions.append(pred_tuple)
            #print(f"Chord: {predicted_chord}, start: {start_time}s, end: {end_time}s")
    return predictions

def print_predictions(preds):
    cleaned_chords = []
    previous_chord = ""
    current_chord_start = 0

    for pred in preds:
        chord = pred[0]
        start = pred[1]
        end = pred[2]
        if previous_chord != chord:
            cleaned_chords.append((chord, current_chord_start, end))
            previous_chord = chord
            current_chord_start = end

    for tuple in cleaned_chords:
        print(f"Chord: {tuple[0]}, start: {tuple[1]}s, end: {tuple[2]}s")

def main():

    # Path to guitar song
    file_path = "../guitar_test_data/D.wav"
    model_path = "best_model.pt"
    play_resampled_audio = True
    if os.path.exists(model_path):
        model = ChordCNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        common_chords = get_common_chords()

        y, sr = load_audio(file_path)
        if play_resampled_audio == True:
            play_audio(y, sr)

        signal_length_sec = len(y) / sr
        qspec = extract_features(y, sr)
        features = create_feature_windows(qspec, signal_length_sec)

        predicted_chords = predict_chords(model, features, common_chords)
        print_predictions(predicted_chords)
        #for i, chord in enumerate(predicted_chords):
        #    print(chord)
    else:
        print("Model not found. Train the model first.")
if __name__ == "__main__":
    main()