from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np

from file_io import get_all_chords, get_common_chords
from data_visualization import plot_chord_distribution

class ChordDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root_dir = root
        self.split = split
        self.features = []
        self.targets = []
        self.chords = []
        self.transform = transform # Transform function -> normalizer
        # Load the data
        self.load_data()

    def _normalize_features(self, feature):
        # Min max scaling to normalize the features between 0 and 1
        min_val = np.min(feature)
        max_val = np.max(feature)
        
        # Avoid division by zero
        if max_val - min_val == 0:
            return feature
        
        return (feature - min_val) / (max_val - min_val)

    def load_data(self):
        datapath = self.root_dir + self.split
        # Store all the files names into a list
        all_file_names = [file for file in os.listdir(datapath)]
        # Loop through the files to read them
        for file_name in all_file_names:
            # Full path to file
            file_path = os.path.join(datapath, file_name)
            
            # Load the .npy file
            data = np.load(file_path, allow_pickle=True)
            
            # Extract features and targets from the loaded data
            features = data['features']
            targets = data['targets']
            chords = data['chords']
            
            # Extend the features and targets lists
            self.features.extend(features)
            self.targets.extend(targets)
            self.chords.extend(chords)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Return feature - target pairs
        if len(self.features) - 1 < idx or len(self.targets) - 1 < idx:
            raise IndexError("Invalid index for features and targets")
        
        feature = self.features[idx]
        target = self.targets[idx]
        chord = self.chords[idx]

        # Apply transform if provided
        feature = self._normalize_features(feature)

        return feature, target, chord
        #return self.features[idx], self.targets[idx]

# For testing purposes
def main():
    
    root = "../data"
    train_split = "/train_serialized"
    test_split = "/test_serialized"
    validation_split = "/validation_serialized"

    train_dataset = ChordDataset(root, train_split)
    test_dataset = ChordDataset(root, test_split)
    validation_dataset = ChordDataset(root, validation_split)

    # Check the lengths
    print("Len train data      :", len(train_dataset))
    print("Len test data       :", len(test_dataset))
    print("Len validation data :", len(validation_dataset))

    # Create a loader
    train_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

    # Test if the testdata is in valid form
    for batch in train_loader:
        features, target, chord = batch
        # Expected shape for features is (batch_size, n_freq_bands, n_time_bands)
        # Expected shape for target is (batch_size, n_common_chords)
        print("feature shape:", features.shape)
        print("target shape :", target.shape)
        print("Chord:", chord)
        break
    
    # allc, trc, tec, vac = get_all_chords()
    # print("All chords in the whole dataset:", allc)
    # print("All chords in the training data:", trc)
    # print("All chords in the testing data:", tec)
    # print("All chords in the val data:", vac)
    # print("ALL COMMON CHORDS:", get_common_chords())

    # Plot bar charts of chord distribution in the data splits
    plot_chord_distribution(train_dataset, "Train", "skyblue")
    plot_chord_distribution(validation_dataset, "Validation", "lightcoral")
    plot_chord_distribution(test_dataset, "Test", "limegreen")
    
if __name__ == "__main__":
    main()