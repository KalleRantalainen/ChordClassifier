# CNN Chord classifier project

## How to run
1. Navigate to folder model
2. Run the process_files.py file by `python process_files.py`. This creates .npz files from the train, test and validation sets. These files contain the feature/target pairs. This also creates 3 new folders inside the data folder for these .npz files.
3. You can run the dataset.py file by `python dataset.py`. This plots the chord distributions and tests if the dataset works.
4. Run train_model.py by `python train_model.py`. This train and tests the model and plots the confusion matrices.
5. You can run inference.py by `python inference.py` to test the model with guitar recordings. Currently it is set to test the model with D.wav file which contains open D chord played with guitar.