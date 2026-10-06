import sys
import random
import statistics
from lib.shared import predict, calculate_error
from itertools import groupby

if __name__=='__main__':
    datasets = ["Iris", "Digits", "Breast Cancer", "Wine"]
    T = int(sys.argv[1])
    # k values obtained from the previous step (lazy-learning.py)
    k = [3, 3, 5, 9]
    
    for dataset_name, k_value in zip(datasets, k):
        # Load the datasets from the files
        Z1_loaded = []
        Z2_loaded = []
        Z3_loaded = []

        with open(f"./datasets/{dataset_name}_Z1.txt", "r") as f:
            for line in f:
                *features, label = map(float, line.strip().split(","))
                Z1_loaded.append((features, int(label)))
        with open(f"./datasets/{dataset_name}_Z2.txt", "r") as f:
            for line in f:
                *features, label = map(float, line.strip().split(","))
                Z2_loaded.append((features, int(label)))
        with open(f"./datasets/{dataset_name}_Z3.txt", "r") as f:
            for line in f:
                *features, label = map(float, line.strip().split(","))
                Z3_loaded.append((features, int(label)))


        for t in range(T):
            # Prediction
            predictions = predict(Z1_loaded, Z2_loaded, k_value)
            true_labels = [t[1] for t in Z2_loaded]
            # Find wrong predictions
            wrong_predictions = [(i, true_class) for i, (predicted_class, true_class) in enumerate(zip(predictions, true_labels)) if predicted_class != true_class]
            
            # Wrong predictions in Z2 must be swapped with same class in Z1
            failed_idx_by_label = {label: [value for _, value in failed_idx] for label, failed_idx in groupby(wrong_predictions, key=lambda x: x[1])}
            for label, failed_idx in failed_idx_by_label.items():
                candidates = [c for c, (_, lbl) in enumerate(Z1_loaded) if lbl == label]
                if len(candidates) < len(failed_idx):
                    print(f"Not enough candidates in Z1 for label {label} to swap with Z2. Skipping swaps for this label.")
                    continue
                random.shuffle(candidates)
                for i, j in zip(failed_idx, candidates):
                    Z1_loaded[j], Z2_loaded[i] = Z2_loaded[i], Z1_loaded[j]

            # for (i, true_class) in wrong_predictions:
            #     candidates = [c for c, (_, label) in enumerate(Z1_loaded) if label == true_class]
            #     j = random.choice(candidates)
            #     Z1_loaded[j], Z2_loaded[i] = Z2_loaded[i], Z1_loaded[j]

        # Generate 20 random subsets of the whole dataset (Z1 + Z2 + Z3) with Z3 length and calculate the error rate for each subset
        subset_size = len(Z3_loaded)
        all_data = Z1_loaded + Z2_loaded + Z3_loaded

        subset_error_rates = []
        for i in range(20):
            subset = random.sample(all_data, subset_size)
            predictions = predict(Z1_loaded, subset, k_value)
            true_labels = [t[1] for t in subset]
            error_rate = calculate_error(predictions, true_labels)
            subset_error_rates.append(error_rate)

        # Calculate the average error rate across all subsets
        average_error_rate = statistics.mean(subset_error_rates)
        std_dev_error_rate = statistics.stdev(subset_error_rates)
        print(f"Dataset: {dataset_name}, Average Error Rate: {average_error_rate:.4f}, Standard Deviation: {std_dev_error_rate:.4f}")
        