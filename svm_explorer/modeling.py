import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def fit_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    C: float,
    gamma_value: float,
    degree: int,
    coef0: float,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma_value,
                    degree=degree,
                    coef0=coef0,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def get_support_vectors_in_original_space(model: Pipeline) -> np.ndarray:
    scaler = model.named_steps["scaler"]
    svc = model.named_steps["svc"]
    return scaler.inverse_transform(svc.support_vectors_)


def get_linear_equation_in_original_space(model: Pipeline):
    svc = model.named_steps["svc"]
    scaler = model.named_steps["scaler"]

    if not hasattr(svc, "coef_"):
        return None, None

    w_scaled = svc.coef_[0]
    b_scaled = svc.intercept_[0]

    w = w_scaled / scaler.scale_
    b = b_scaled - np.sum((w_scaled * scaler.mean_) / scaler.scale_)

    return w, b