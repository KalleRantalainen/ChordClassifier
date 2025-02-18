from torch.utils.data import Dataset

class ChordDataset(Dataset):
    def __init__(self, root, split):
        self.root_dir = root
        self.split = split
        self.features = []
        self.targets = []

    def load_data(self):
        pass

    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        return None, None
    
# For testing purposes
def main():
    pass
if __name__ == "__main__":
    main()