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

def get_chords_and_times(file_name):
    """
    Get all the chords and chord timestaps for a song from the json file.
    """
    json_file_loc = "../data/file_descriptions.json"

    # Load the JSON data
    with open(json_file_loc, "r") as f:
        data = json.load(f)

    # Loop through the data to find the matching file
    for entry in data:
        if entry.get("location") == file_name:
            # Extract chords and chord times
            chords = entry.get("chords", [])
            chords_time = entry.get("chords_time", [])
            return chords, chords_time

    # If not found, return None
    print(f"File '{file_name}' not found in JSON.")
    return None, None


def get_all_file_names(folder):
    """
    Get all file names from the data split defined by folder.
    So from test, train or validation
    """
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
    """
    Get all chords present in the data.
    """
    json_path = "../data/file_descriptions.json"
    all_chords = []
    train_chords = []
    test_chords = []
    validation_chords = []
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        # Loop through the json dicts
        for item in data:
            # Check if 'chords' key is in the dict
            if "chords" in item:
                # Get all the chords and append the ones we don't yet have
                if item['location'].split("/")[0] == "train":
                    for chord in item['chords']:
                        if chord not in train_chords:
                            train_chords.append(chord)
                elif item['location'].split("/")[0] == "test":
                    for chord in item['chords']:
                        if chord not in test_chords:
                            test_chords.append(chord)
                elif item['location'].split("/")[0] == "validation":
                    for chord in item['chords']:
                        if chord not in validation_chords:
                            validation_chords.append(chord)
                for chord in item['chords']:
                    if chord not in all_chords:
                        all_chords.append(chord)
    return all_chords, train_chords, test_chords, validation_chords


def get_common_chords():
    """
    Get chords that are present in all of the data splits.
    """
    _, train_chords, test_chords, validation_chords = get_all_chords()

    val_chords_in_train = []
    for val_ch in validation_chords:
        if val_ch in train_chords:
            val_chords_in_train.append(val_ch)

    common_chords = []
    for test_ch in test_chords:
        if test_ch in val_chords_in_train:
            common_chords.append(test_ch)

    return common_chords

# For testing
def main():

    # Get all chords in the data + chords in each data split
    all_chords, train_chords, test_chords, validation_chords = get_all_chords()
    # Get chords that all present in every data split
    common_chords = get_common_chords()

    print("len all", len(all_chords))
    print("len train", len(train_chords))
    print("len test", len(test_chords))
    print("len validation", len(validation_chords))
    print("len common chords", len(common_chords))

if __name__ == "__main__":
    main()