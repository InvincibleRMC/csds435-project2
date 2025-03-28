import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import rand_score


def main() -> None:
    np.random.seed(123)

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cho", "iyer"], default="cho")
    parser.add_argument(
        "--algorithm",
        choices=[
            "K_Means",
            "Density_Based",
            "Spectral",
        ],
        default="K_Means",
    )

    args = parser.parse_args()

    if args.dataset == "cho":
        filename = "cho.txt"
        class_count = 5
    else:
        filename = "iyer.txt"
        class_count = 10

    match args.algorithm:
        case "K_Means":
            algorithm_type = 1
        case "Density_Based":
            algorithm_type = 2
        case "Spectral":
            algorithm_type = 3
        case _:
            raise Exception()

    f = open(os.path.join("data", filename))

    data = np.loadtxt(f)

    ACTUAL_LOCATION = 1

    X = data[:, (ACTUAL_LOCATION + 1) : data.T.size]
    Y = data[:, ACTUAL_LOCATION]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=0,
    )

    X_train_norm = preprocessing.normalize(X_train)
    X_test_norm = preprocessing.normalize(X_test)

    kmeans = KMeans(n_clusters=class_count, random_state=0, n_init="auto")
    kmeans.fit(X_train_norm)

    predicted_y_test = kmeans.predict(X_test_norm)

    print(f"Rand Index for {filename} is {rand_score(y_test, predicted_y_test)}")

    N_COMPONENTS = 2
    # Apply PCA
    pca = PCA(n_components=N_COMPONENTS)
    pca_result = pca.fit_transform(X_train_norm)

    pca_x = pca_result[:, 0]
    pca_y = pca_result[:, 1]

    plt.figure()
    sns.scatterplot(x=pca_x, y=pca_y, hue=kmeans.labels_)
    plt.show()


if __name__ == "__main__":
    main()
