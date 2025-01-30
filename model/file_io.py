# Load .wav files and serialize data using pickle.
# Get all the different chords present in the dataset

import librosa
import json

def read_wav_file(path: str):
    y, sr = librosa.load(path, sr=None)
    return sr, y

def serialize_data():
    #TODO: Pickle
    pass

def get_all_file_names(folder):
    json_path = "../data/file_descriptions.json"
    all_files = []
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        for item in data:
            if "location" in item:
                # Check if the file is test/train/validation
                if item['location'].split("/")[0] == folder:
                    file_loc = "../data/" + item['location']
                    all_files.append(file_loc)
    return all_files

def get_all_chords():
    json_path = "../data/file_descriptions.json"
    all_chords = []
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        # Loop through the json dicts
        for item in data:
            # Check if 'chords' key is in the dict
            if "chords" in item:
                # Get all the chords and append the ones we don't yet have
                for chord in item['chords']:
                    if chord not in all_chords:
                        all_chords.append(chord)
    return all_chords