import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from torchvision import transforms
from data_visualization import plot_confusion_matrix

from dataset import ChordDataset
from cnn import ChordCNN

def train(train_ds, test_ds, val_ds):
    # Define the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define some hyper params for training
    epochs = 20
    learning_rate = 1e-4
    batch_size = 32

    # Define the CNN model
    model = ChordCNN()

    # Pass model to the available device.
    model = model.to(device)

    # Define the optimizer and give the parameters of the CNN model to an optimizer.
    optimizer = Adam(params=model.parameters(), lr=learning_rate)

    # Using CrossEntropyLoss
    loss_function = torch.nn.CrossEntropyLoss()

    # Define the data loaders
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

    # Define early stopping parameters
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 5
    patience_counter = 0
    best_model = None

    for epoch in range(epochs):

        # Lists to hold the corresponding losses of each epoch.
        epoch_loss_training = []
        epoch_loss_validation = []

        # Indicate that we are in training mode, so (e.g.) dropout
        # will function
        model.train()

        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        # For each batch of our dataset.
        for batch in train_loader_tqdm:
            # Zero the gradient of the optimizer.
            optimizer.zero_grad()

            # Get the batches.
            x, y, _ = batch

            # Give them to the appropriate device.
            x = x.to(device)
            y = y.to(device).float() # Convert to float format -> cross entropy wants this

            # Get the predictions.
            y_hat = model(x)

            # Calculate the loss .
            loss = loss_function(y_hat, y)

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Append the loss of the batch
            epoch_loss_training.append(loss.item())

            # Update progress bar
            train_loader_tqdm.set_postfix(loss=loss.item())

        # Indicate that we are in evaluation mode
        model.eval()

        valid_loader_tqdm = tqdm(validation_loader, desc=f"Validating Epoch {epoch+1}")

        # Evaluate the model using validation data
        with torch.no_grad():
            # For every batch of our validation data.
            for batch in valid_loader_tqdm:
                # Get the batch
                x_val, y_val, _ = batch

                # Pass the data to the appropriate device.
                x_val = x_val.to(device)
                y_val = y_val.to(device).float()

                # Get the predictions of the model.
                y_hat = model(x_val)

                # Calculate the loss.
                loss = loss_function(y_hat, y_val)

                # Append the validation loss.
                epoch_loss_validation.append(loss.item())
                # Update the progress bar
                valid_loader_tqdm.set_postfix(loss=loss.item())

        # Calculate mean losses.
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()
        print(f"Epoch {epoch} training loss  : {epoch_loss_training:.2f}")
        print(f"Epoch {epoch} validation loss: {epoch_loss_validation:.2f}")

        # Check early stopping conditions.
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(model.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1

        # If we have to stop, do the testing.
        if patience_counter >= patience or epoch == epochs - 1:
            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            if best_model is None:
                print('No best model. ')
            else:
                # Save and load the best model.
                torch.save(best_model, 'best_model.pt')
                model.load_state_dict(best_model)

                # Process similar to validation.
                print('Starting testing', end=' | ')
                testing_loss = []
                labels_test = []
                predictions_test = []
                model.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        x_test, y_test, _ = batch

                        # Pass the data to the appropriate device.
                        x_test = x_test.to(device)
                        y_test = y_test.to(device).float()

                        # make the prediction
                        y_hat = model(x_test)

                        # Calculate the loss.
                        loss = loss_function(y_hat, y_test)
                        testing_loss.append(loss.item())

                        # Save the predictions and labels for later analysis.
                        predictions_test.append(y_hat.argmax(dim=-1).cpu().numpy())
                        labels_test.append(y_test.argmax(dim=-1).cpu().numpy())

                testing_loss = np.array(testing_loss).mean()
                print(f'Testing loss: {testing_loss:7.4f}')

                # Calculate the accuracy.
                predictions_test = np.concatenate(predictions_test)
                labels_test = np.concatenate(labels_test)

                # Calculate the amount of correct predictions
                correct_predictions = np.equal(predictions_test, labels_test)
                accuracy = np.mean(correct_predictions)
                print("Amount of correct predictions:", np.sum(correct_predictions))
                print("Test accuracy:", accuracy)

                # Plot confusiuon matrix from the predictions.
                plot_confusion_matrix(labels_test, predictions_test, test_loader)
                break

# Train the model
def main():
    
    root = "../data"
    train_split = "/train_serialized"
    test_split = "/test_serialized"
    validation_split = "/validation_serialized"

    train_dataset = ChordDataset(root, train_split)
    test_dataset = ChordDataset(root, test_split)
    validation_dataset = ChordDataset(root, validation_split)

    train(train_dataset, test_dataset, validation_dataset)
    
if __name__ == "__main__":
    main()