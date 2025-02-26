from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from file_io import get_common_chords

def plot_chord_distribution(dataset, dataset_name, color):
    """Plots the chord distribution for a given dataset."""
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    all_chords = []
    for _, _, chord in data_loader:
        all_chords.extend(chord)

    # Count occurrences of each chord
    chord_counts = Counter(all_chords)
    print(f"TOTAL CHORD COUNT ({dataset_name}):", len(all_chords))
    print(f"UNIQUE HORD COUNT ({dataset_name}): {len(set(all_chords))}")
    print(f"ALL CHORDS IN {dataset_name}:")
    print(set(all_chords))

    # Plot the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(chord_counts.keys(), chord_counts.values(), color=color)
    plt.xlabel("Chord Labels")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Chords in the {dataset_name} Dataset")
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def plot_confusion_matrix(labels_test, predictions_test, test_loader):
    chord_list = get_common_chords()

    # Create confusion matrix from the predicted and gt labels
    cm = confusion_matrix(labels_test, predictions_test)

    print("PREDICTION TEST:", predictions_test, "MAX VAL:", np.max(predictions_test))
    print("LABELS TEST:", labels_test, "MAX VAL:", np.max(predictions_test))
    print("CHORD LIST:", chord_list)

    # Plot confusion matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=chord_list, yticklabels=chord_list)
    plt.xlabel("Predicted Chord")
    plt.ylabel("True Chord")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()