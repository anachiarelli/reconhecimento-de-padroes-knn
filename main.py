import os
import sys
import statistics
import numpy as np
from sklearn.datasets import load_iris, load_digits, load_breast_cancer

def euclidean_distance(a:list[float], b:list[float]) -> float:
    sum = 0.0
    for x_i, y_i in zip(a, b):
        sum += (x_i - y_i) ** 2
    return sum ** 0.5

def predict(base, target, k):
    base_X = [b[0] for b in base]
    base_y = [b[1] for b in base]

    target_X = [t[0] for t in target]

    predictions = []
    for t in target_X:
        distances_by_class = [(euclidean_distance(t, b), b_class) for b, b_class in zip(base_X, base_y)]
        distances_by_class.sort()
        k_nearest_neighbors = distances_by_class[:k]

        predicted_class = statistics.mode([neighbor[1] for neighbor in k_nearest_neighbors])
        predictions.append(predicted_class)

    return predictions

def calculate_error(predictions, true_labels):
    errors = 0
    for predicted_class, true_class in zip(predictions, true_labels):
        if predicted_class != true_class:
            errors += 1
    return errors / len(true_labels)

if __name__=='__main__':
    r1, r2, r3 = map(float, sys.argv[1:4])

    loaders = [
        ("Iris", load_iris),
        # ("Digits", load_digits),
        ("Breast Cancer", load_breast_cancer)
    ]

    for dataset_name, loader in loaders:
        dataset = loader()
        X, y = dataset.data, dataset.target

        # Divide the dataset into classes
        data_by_target = {}
        for data, target in zip(X, y):
            if target not in data_by_target:
                data_by_target[target] = []
            data_by_target[target].append(data)

        # Split the data into three parts based on the specified ratios (stratified sampling)
        Z1, Z2, Z3 = [], [], []
        for target, data_points in data_by_target.items():
            n = len(data_points)
            n1 = int(n * r1)
            n2 = int(n * r2)
            n3 = n - n1 - n2

            Z1.extend([(d, target) for d in data_points[:n1]])
            Z2.extend([(d, target) for d in data_points[n1:n1+n2]])
            Z3.extend([(d, target) for d in data_points[n1+n2:]])

        # save Z1, Z2, Z3 to files
        os.makedirs("./datasets", exist_ok=True)
        with open(f"./datasets/{dataset_name}_Z1.txt", "w") as f:
            for data, target in Z1:
                f.write(",".join(map(str, data)) + f",{target}\n")
        with open(f"./datasets/{dataset_name}_Z2.txt", "w") as f:
            for data, target in Z2:
                f.write(",".join(map(str, data)) + f",{target}\n")
        with open(f"./datasets/{dataset_name}_Z3.txt", "w") as f:
            for data, target in Z3:
                f.write(",".join(map(str, data)) + f",{target}\n")

        # Load the datasets from the files
        Z1_loaded = []
        Z2_loaded = []
        Z3_loaded = []

        with open(f"./datasets/{dataset_name}_Z1.txt", "r") as f:
            for line in f:
                *data, target = map(float, line.strip().split(","))
                Z1_loaded.append((data, int(target)))
        with open(f"./datasets/{dataset_name}_Z2.txt", "r") as f:
            for line in f:
                *data, target = map(float, line.strip().split(","))
                Z2_loaded.append((data, int(target)))
        with open(f"./datasets/{dataset_name}_Z3.txt", "r") as f:
            for line in f:
                *data, target = map(float, line.strip().split(","))
                Z3_loaded.append((data, int(target)))


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

        # Divide Z3 into 20 random subsets and calculate the error rate for each subset using the best k
        subset_size = len(Z3_loaded) // 20
        subset_error_rates = []
        for i in range(20):
            subset = Z3_loaded[i*subset_size:(i+1)*subset_size]
            predictions = predict(Z1_loaded, subset, best_k)
            true_labels = [t[1] for t in subset]
            subset_error_rates.append(calculate_error(predictions, true_labels))

        # Calculate the average error rate across all subsets
        average_error_rate = statistics.mean(subset_error_rates)
        std_dev_error_rate = statistics.stdev(subset_error_rates)
        print(f"Dataset: {dataset_name}, Best k: {best_k}, Average Error Rate: {average_error_rate:.4f}, Standard Deviation: {std_dev_error_rate:.4f}")