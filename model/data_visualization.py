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
    unique_chords = set(all_chords)

    print(f"TOTAL CHORD COUNT ({dataset_name}):", len(all_chords))
    print(f"UNIQUE HORD COUNT ({dataset_name}): {len(unique_chords)}")
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

    # Compute statistics to get rid of useless chords
    chord_count_values = list(chord_counts.values())
    average_count = np.mean(chord_count_values)
    median_count = np.median(chord_count_values)
    percentile_90 = np.percentile(chord_count_values, 10)  # 10th percentile means 90% are above
    
    # Find the maximum chord count
    max_count = max(chord_counts.values())
    
    # Compute 10% of the max count. So each chord should have atleast 10% of maximum count so
    # we do not have huge class imbalances in the dataset
    threshold_10_percent = max_count * 0.10
    
    # Find chords below this 10% threshold
    below_threshold_chords = {chord: count for chord, count in chord_counts.items() if count < threshold_10_percent}
    
    # Print results in a readable format
    print("\n" + "=" * 50)
    print(f"Chord Distribution Statistics for {dataset_name} Dataset")
    print("=" * 50)
    print(f"Total chord occurrences: {len(all_chords)}")
    print(f"Unique chord count: {len(unique_chords)}")
    print(f"Average chord count: {average_count:.2f}")
    print(f"Median chord count: {median_count}")
    print(f"90% of chords have at least {int(percentile_90)} occurrences (10% have fewer).")
    print(f"Most frequent chord count: {max_count}")
    print(f"10% of max count: {int(threshold_10_percent)}")
    print(f"Chords appearing less than {int(threshold_10_percent)} times: {len(below_threshold_chords)}")
    print("Rare Chords (Below 10% Threshold):")
    for chord, count in below_threshold_chords.items():
        print(f"  {chord}: {count} occurrences")
    print("=" * 50 + "\n")

def plot_confusion_matrix(labels_test, predictions_test, test_loader):
    chord_list = get_common_chords()

    # Compute confusion matrix
    cm_raw = confusion_matrix(labels_test, predictions_test)

    # Compute accuracy for each chord separately
    per_class_accuracy = cm_raw.diagonal() / cm_raw.sum(axis=1)

    # Normalize by row (i.e., per-class percentages)
    cm_norm = cm_raw.astype('float') / cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero for empty rows

    print("PREDICTION TEST:", predictions_test, "MAX VAL:", np.max(predictions_test))
    print("LABELS TEST:", labels_test, "MAX VAL:", np.max(labels_test))
    print("CHORD LIST:", chord_list)
    print("Len chords:", len(chord_list))

    # Set up the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw confusion matrix
    sns.heatmap(cm_raw, annot=False, fmt="d", cmap="Blues", xticklabels=chord_list, yticklabels=chord_list, ax=axes[0])
    axes[0].set_title("Raw Confusion Matrix")
    axes[0].set_xlabel("Predicted Chord")
    axes[0].set_ylabel("True Chord")
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].tick_params(axis="y", rotation=0)

    # Normalized confusion matrix
    sns.heatmap(cm_norm, annot=False, fmt=".2f", cmap="Blues", xticklabels=chord_list, yticklabels=chord_list, ax=axes[1])
    axes[1].set_title("Normalized Confusion Matrix (Per-Class Percentage)")
    axes[1].set_xlabel("Predicted Chord")
    axes[1].set_ylabel("True Chord")
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].tick_params(axis="y", rotation=0)

    #for i, chord in enumerate(chord_list):
    #    print(f"Accuracy for {chord}: {per_class_accuracy[i]:.2f}")

    # Adjust layout
    plt.tight_layout()
    plt.show()