import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons


def make_dataset(dataset_name: str, n_samples: int, noise: float, seed: int):
    rng = np.random.RandomState(seed)

    if dataset_name == "Lineárisan szeparálható":
        cluster_std = 0.55 + 1.1 * noise
        X, y = make_blobs(
            n_samples=n_samples,
            centers=[(-2.2, -1.7), (2.0, 1.7)],
            cluster_std=cluster_std,
            random_state=seed,
        )

    elif dataset_name == "Átfedő klaszterek":
        cluster_std = 1.15 + 1.4 * noise
        X, y = make_blobs(
            n_samples=n_samples,
            centers=[(-1.2, -1.0), (1.35, 1.15)],
            cluster_std=cluster_std,
            random_state=seed,
        )

    elif dataset_name == "Két félhold":
        X, y = make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=seed,
        )

    elif dataset_name == "Koncentrikus körök":
        X, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=0.45,
            random_state=seed,
        )

    elif dataset_name == "XOR":
        X = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
        X = X + rng.normal(scale=noise * 0.9, size=X.shape)
        y = ((X[:, 0] * X[:, 1]) > 0).astype(int)

    else:
        raise ValueError(f"Ismeretlen dataset: {dataset_name}")

    return X.astype(float), y.astype(int)