import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.preprocessing import StandardScaler

from .data import make_dataset
from .modeling import get_support_vectors_in_original_space


def make_mesh(X: np.ndarray, n_grid: int = 240, padding: float = 1.25):
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)

    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def build_decision_figure(
    X: np.ndarray,
    y: np.ndarray,
    model,
    title: str,
    show_regions: bool = True,
    show_boundary: bool = True,
    show_margins: bool = True,
    show_support_vectors: bool = True,
    height: int = 620,
):
    xx, yy, grid = make_mesh(X)
    pred = model.predict(grid).reshape(xx.shape)
    decision = model.decision_function(grid).reshape(xx.shape)

    support_vectors = get_support_vectors_in_original_space(model)

    fig = go.Figure()

    if show_regions:
        fig.add_trace(
            go.Contour(
                x=xx[0, :],
                y=yy[:, 0],
                z=pred,
                showscale=False,
                hoverinfo="skip",
                opacity=0.30,
                colorscale=[
                    [0.0, "rgb(130, 180, 255)"],
                    [0.499, "rgb(130, 180, 255)"],
                    [0.5, "rgb(255, 165, 165)"],
                    [1.0, "rgb(255, 165, 165)"],
                ],
                contours=dict(
                    start=0,
                    end=1,
                    size=1,
                    coloring="fill",
                    showlines=False,
                ),
                line=dict(width=0),
                name="Döntési régiók",
            )
        )

    mask0 = y == 0
    mask1 = y == 1

    fig.add_trace(
        go.Scatter(
            x=X[mask0, 0],
            y=X[mask0, 1],
            mode="markers",
            name="0. osztály",
            marker=dict(
                size=9,
                color="rgb(37, 99, 235)",
                line=dict(color="white", width=1),
            ),
            hovertemplate="x₁=%{x:.2f}<br>x₂=%{y:.2f}<extra>0. osztály</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X[mask1, 0],
            y=X[mask1, 1],
            mode="markers",
            name="1. osztály",
            marker=dict(
                size=9,
                color="rgb(220, 38, 38)",
                line=dict(color="white", width=1),
            ),
            hovertemplate="x₁=%{x:.2f}<br>x₂=%{y:.2f}<extra>1. osztály</extra>",
        )
    )

    if show_boundary:
        fig.add_trace(
            go.Contour(
                x=xx[0, :],
                y=yy[:, 0],
                z=decision,
                showscale=False,
                hoverinfo="skip",
                contours=dict(
                    start=0.0,
                    end=0.0,
                    size=1.0,
                    coloring="lines",
                    showlabels=False,
                ),
                line=dict(color="black", width=3),
                name="Döntési határ",
            )
        )

    if show_margins:
        for level, label in [(-1.0, "Margin (-1)"), (1.0, "Margin (+1)")]:
            fig.add_trace(
                go.Contour(
                    x=xx[0, :],
                    y=yy[:, 0],
                    z=decision,
                    showscale=False,
                    hoverinfo="skip",
                    contours=dict(
                        start=level,
                        end=level,
                        size=1.0,
                        coloring="lines",
                        showlabels=False,
                    ),
                    line=dict(color="gray", width=2),
                    name=label,
                )
            )

    if show_support_vectors:
        fig.add_trace(
            go.Scatter(
                x=support_vectors[:, 0],
                y=support_vectors[:, 1],
                mode="markers",
                name="Support vectorok",
                marker=dict(
                    size=18,
                    color="rgba(255,255,255,0)",
                    line=dict(color="black", width=2),
                    symbol="circle",
                ),
                hovertemplate="x₁=%{x:.2f}<br>x₂=%{y:.2f}<extra>Support vector</extra>",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        dragmode="pan",
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        uirevision="keep",
    )

    fig.update_xaxes(title_text="x₁")
    fig.update_yaxes(title_text="x₂", scaleanchor="x", scaleratio=1)

    return fig


def build_kernel_matrix_figure(
    X: np.ndarray,
    kernel_name: str,
    gamma_value: float,
    degree: int,
    coef0: float,
    n_show: int = 50,
):
    X_small = X[:n_show]
    X_small = StandardScaler().fit_transform(X_small)

    if kernel_name == "linear":
        K = linear_kernel(X_small)
    elif kernel_name == "poly":
        K = polynomial_kernel(X_small, gamma=gamma_value, degree=degree, coef0=coef0)
    else:
        K = rbf_kernel(X_small, gamma=gamma_value)

    fig = go.Figure(
        data=go.Heatmap(
            z=K,
            colorscale="Viridis",
            colorbar=dict(title="K(xᵢ, xⱼ)"),
            hovertemplate="i=%{y}<br>j=%{x}<br>K=%{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Kernel-mátrix ({kernel_name})",
        template="plotly_white",
        height=460,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    fig.update_xaxes(title="Minták indexe")
    fig.update_yaxes(title="Minták indexe")

    return fig


def build_explicit_kernel_trick_figures(n_samples: int, noise: float, seed: int):
    X, y = make_dataset("Koncentrikus körök", n_samples=n_samples, noise=noise, seed=seed)

    z = X[:, 0] ** 2 + X[:, 1] ** 2
    z0 = z[y == 0]
    z1 = z[y == 1]
    threshold = 0.5 * (z0.mean() + z1.mean())

    fig2d = go.Figure()
    fig2d.add_trace(
        go.Scatter(
            x=X[y == 0, 0],
            y=X[y == 0, 1],
            mode="markers",
            name="0. osztály",
            marker=dict(size=8, color="rgb(37, 99, 235)", line=dict(color="white", width=1)),
        )
    )
    fig2d.add_trace(
        go.Scatter(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            mode="markers",
            name="1. osztály",
            marker=dict(size=8, color="rgb(220, 38, 38)", line=dict(color="white", width=1)),
        )
    )

    theta = np.linspace(0, 2 * np.pi, 400)
    r = np.sqrt(max(threshold, 1e-8))
    circle_x = r * np.cos(theta)
    circle_y = r * np.sin(theta)

    fig2d.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            name="x₁² + x₂² = konstans",
            line=dict(color="black", width=3),
        )
    )

    fig2d.update_layout(
        title="Eredeti 2D tér – kör alakú elválasztás",
        template="plotly_white",
        height=520,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", y=1.02, x=0.0),
    )
    fig2d.update_xaxes(title="x₁")
    fig2d.update_yaxes(title="x₂", scaleanchor="x", scaleratio=1)

    fig3d = go.Figure()
    fig3d.add_trace(
        go.Scatter3d(
            x=X[y == 0, 0],
            y=X[y == 0, 1],
            z=z[y == 0],
            mode="markers",
            name="0. osztály",
            marker=dict(size=4.5, color="rgb(37, 99, 235)"),
        )
    )
    fig3d.add_trace(
        go.Scatter3d(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            z=z[y == 1],
            mode="markers",
            name="1. osztály",
            marker=dict(size=4.5, color="rgb(220, 38, 38)"),
        )
    )

    gx = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 25)
    gy = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 25)
    XX, YY = np.meshgrid(gx, gy)
    ZZ = np.full_like(XX, threshold)

    fig3d.add_trace(
        go.Surface(
            x=XX,
            y=YY,
            z=ZZ,
            opacity=0.45,
            showscale=False,
            name="Elválasztó sík",
        )
    )

    fig3d.update_layout(
        title="Emelt 3D tér – síkkal elválasztható",
        template="plotly_white",
        height=520,
        margin=dict(l=10, r=10, t=55, b=10),
        scene=dict(
            xaxis_title="x₁",
            yaxis_title="x₂",
            zaxis_title="x₁² + x₂²",
        ),
    )

    return fig2d, fig3d, threshold