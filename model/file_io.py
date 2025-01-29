# Load .wav files and serialize data using pickle.
# Get all the different chords present in the dataset

import librosa
import json

def read_wav_file(folder: str):
    path = "../data/" + folder
    # TODO: read file
    #return fs, y

def serialize_data():
    #TODO: Pickle
    pass

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