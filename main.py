import os
import sys
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine

if __name__=='__main__':
    r1, r2, r3 = map(float, sys.argv[1:4])

    loaders = [
        ("Iris", load_iris),
        ("Digits", load_digits),
        ("Breast Cancer", load_breast_cancer),
        ("Wine", load_wine)
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
