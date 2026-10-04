import sys
import sklearn
import numpy as np
from sklearn.datasets import load_iris, load_digits, load_breast_cancer

r1, r2, r3 = map(float, sys.argv[1:4])
print(r1, r2, r3)

loaders = [
    ("Iris", load_iris),
    ("Digits", load_digits),
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

    print(f"Dataset: {dataset_name}")
    print(f"Z1: {len(Z1)} samples")
    print(f"Z2: {len(Z2)} samples")
    print(f"Z3: {len(Z3)} samples")
    