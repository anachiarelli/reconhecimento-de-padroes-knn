import statistics
import random
from lib.shared import predict, calculate_error

if __name__=='__main__':

    datasets = ["Iris", "Digits", "Breast Cancer", "Wine"]

    for dataset_name in datasets:
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


        # LAZY LEARNING
        # Predict the classes of Z2 using Z1 as the base dataset
        min_k = 3
        max_k = 15
        error_rates = []

        for k in range (min_k, max_k, 2):
            predictions = predict(Z1_loaded, Z2_loaded, k)
            true_labels = [t[1] for t in Z2_loaded]
            error_rates.append(calculate_error(predictions, true_labels))

        # get the first index of the minimum error rate
        min_error_index = 0
        for i in range(1, len(error_rates)):
            if error_rates[i] < error_rates[min_error_index]:
                min_error_index = i

        # get the k with the minimum error rate
        best_k = min_k + min_error_index * 2
        print(best_k)

        # Generate 20 random subsets of the whole dataset (Z1 + Z2 + Z3) with Z3 length and calculate the error rate for each subset
        subset_size = len(Z3_loaded)
        all_data = Z1_loaded + Z2_loaded + Z3_loaded

        subset_error_rates = []
        for i in range(20):
            subset = random.sample(all_data, subset_size)
            predictions = predict(Z1_loaded, subset, best_k)
            true_labels = [t[1] for t in subset]
            error_rate = calculate_error(predictions, true_labels)
            subset_error_rates.append(error_rate)

        # Calculate the average error rate across all subsets
        average_error_rate = statistics.mean(subset_error_rates)
        std_dev_error_rate = statistics.stdev(subset_error_rates)
        print(f"Dataset: {dataset_name}, Average Error Rate: {average_error_rate:.4f}, Standard Deviation: {std_dev_error_rate:.4f}")
