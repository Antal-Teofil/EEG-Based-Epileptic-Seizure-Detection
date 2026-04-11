import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# Oldal / stílus
# ============================================================
st.set_page_config(
    page_title="SVM Explorer – Interaktív vizualizáció",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        padding-top: 1.2rem;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #111827 55%, #1d4ed8 100%);
        padding: 1.4rem 1.6rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.15;
    }
    .hero p {
        margin-top: 0.65rem;
        color: rgba(255,255,255,0.92);
        font-size: 1rem;
    }
    .card {
        background: #0f172a08;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }
    .card h3 {
        margin-top: 0;
        margin-bottom: 0.4rem;
    }
    .pillrow {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.6rem;
    }
    .pill {
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        background: rgba(29, 78, 216, 0.10);
        border: 1px solid rgba(29, 78, 216, 0.18);
        font-size: 0.92rem;
    }
    .formula {
        background: #111827;
        color: #f9fafb;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>SVM Explorer</h1>
        <p>
            Interaktív, magyar nyelvű felület a Support Vector Machine működésének
            szemléltetésére: döntési határ, margin, support vectorok, kernel-trükk,
            lineáris és nemlineáris szeparáció.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Segéd konstansok
# ============================================================
STAGES = [
    "1. Csak az adatpontok",
    "2. Döntési régiók",
    "3. Döntési határ",
    "4. Margin + support vectorok",
]

DATASETS = [
    "Lineárisan szeparálható",
    "Átfedő klaszterek",
    "Két félhold",
    "Koncentrikus körök",
    "XOR",
]

KERNELS = ["linear", "poly", "rbf"]

if "seed" not in st.session_state:
    st.session_state.seed = 42


# ============================================================
# Adatkészletek
# ============================================================
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


# ============================================================
# Modell
# ============================================================
def fit_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    C: float,
    gamma_value: float,
    degree: int,
    coef0: float,
):
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


# ============================================================
# Plot segédfüggvények
# ============================================================
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
    model: Pipeline,
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

    # Explicit feature map:
    # φ(x1, x2) = (x1, x2, x1² + x2²)
    z = X[:, 0] ** 2 + X[:, 1] ** 2
    z0 = z[y == 0]
    z1 = z[y == 1]
    threshold = 0.5 * (z0.mean() + z1.mean())

    # 2D eredeti tér
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

    # 3D emelt tér
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


def stage_explanation(stage_text: str, kernel: str) -> str:
    if stage_text.startswith("1."):
        return (
            "Most csak a pontokat látod. Itt még nincs megmutatva, "
            "hogyan választja el őket az SVM."
        )
    if stage_text.startswith("2."):
        return (
            "A háttér már a modell által becsült döntési régiókat mutatja: "
            "a sík mely részeit melyik osztályhoz rendeli."
        )
    if stage_text.startswith("3."):
        return (
            "A fekete görbe/vonal a döntési határ: itt vált a modell egyik "
            "osztályról a másikra."
        )
    if kernel == "linear":
        return (
            "A szürke vonalak a marginok, a kiemelt pontok a support vectorok. "
            "Ezek határozzák meg a maximális margójú elválasztót."
        )
    return (
        "Nemlineáris kernel esetén a döntési határ görbült lehet, de a support "
        "vectorok itt is kulcsszerepet játszanak."
    )


def kernel_label(kernel: str) -> str:
    return {"linear": "lineáris", "poly": "polinomiális", "rbf": "RBF"}[kernel]


def short_kernel_explanation(kernel: str) -> str:
    if kernel == "linear":
        return "Jó választás, ha az adatok közel egyenessel elválaszthatók."
    if kernel == "poly":
        return "Polinomiális kölcsönhatásokat is figyelembe vesz, ezért görbült határokat tud tanulni."
    return "Nagyon rugalmas, lokális hasonlóság alapján görbült döntési határt épít."


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Globális beállítások")

    dataset_name = st.selectbox("Adathalmaz", DATASETS, index=2)

    kernel = st.selectbox(
        "Fő kernel",
        KERNELS,
        index=2,
        format_func=kernel_label,
    )

    c_log10 = st.slider("log10(C)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    C = float(10 ** c_log10)

    gamma_log10 = st.slider("log10(gamma)", min_value=-2.0, max_value=2.0, value=-0.1, step=0.1)
    gamma_value = float(10 ** gamma_log10)

    degree = st.slider("Polinom fokszám (degree)", min_value=2, max_value=6, value=3, step=1)
    coef0 = st.slider("Polinom offset (coef0)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    n_samples = st.slider("Minták száma", min_value=40, max_value=320, value=140, step=10)

    default_noise = 0.15 if dataset_name in ["Két félhold", "Koncentrikus körök", "XOR"] else 0.10
    noise = st.slider("Zaj", min_value=0.0, max_value=0.70, value=default_noise, step=0.01)

    if st.button("Új minta ugyanilyen beállításokkal"):
        st.session_state.seed += 1

    st.number_input("Seed", min_value=0, max_value=10000, step=1, key="seed")

    st.markdown("---")
    st.markdown(
        """
        **Gyorstipp**
        - lineáris klaszterekhez: `linear`
        - félhold / kör / XOR esetén: `rbf` vagy `poly`
        - nagyobb `C`: szigorúbb illesztés
        - nagyobb `gamma`: lokálisabb, kacskaringósabb határ
        """
    )


# ============================================================
# Fő adat + modell
# ============================================================
X, y = make_dataset(
    dataset_name=dataset_name,
    n_samples=n_samples,
    noise=noise,
    seed=int(st.session_state.seed),
)

main_model = fit_svm(
    X=X,
    y=y,
    kernel=kernel,
    C=C,
    gamma_value=gamma_value,
    degree=degree,
    coef0=coef0,
)

y_pred = main_model.predict(X)
train_acc = accuracy_score(y, y_pred)

svc = main_model.named_steps["svc"]
n_support = int(len(svc.support_))
support_ratio = n_support / len(X)

w, b = get_linear_equation_in_original_space(main_model)


# ============================================================
# Fejléc kártyák
# ============================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="card">
            <h3>Mi az SVM?</h3>
            <div>
                Egy osztályozó módszer, amely olyan döntési határt keres,
                ami a lehető legnagyobb margóval választja el az osztályokat.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="card">
            <h3>Mi a margin?</h3>
            <div>
                A döntési határ és a legközelebbi pontok közti „biztonsági sáv”.
                Az SVM ezt igyekszik maximalizálni.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="card">
            <h3>Kik a support vectorok?</h3>
            <div>
                Azok a pontok, amelyek a határhoz legközelebb vannak, és ténylegesen
                meghatározzák a modell helyzetét.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="pillrow">
        <div class="pill">Adathalmaz: {dataset_name}</div>
        <div class="pill">Kernel: {kernel_label(kernel)}</div>
        <div class="pill">C = {C:.4f}</div>
        <div class="pill">gamma = {gamma_value:.4f}</div>
        <div class="pill">Pontok száma = {len(X)}</div>
        <div class="pill">Train accuracy = {100 * train_acc:.1f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Tabok
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Alapötlet",
        "Kernel-trükk",
        "Kernel-összehasonlítás",
        "Hiperaparaméterek hatása",
    ]
)


# ============================================================
# TAB 1 – Alapötlet
# ============================================================
with tab1:
    stage = st.select_slider("Megjelenítési lépés", options=STAGES, value=STAGES[3])

    stage_num = STAGES.index(stage) + 1
    show_regions = stage_num >= 2
    show_boundary = stage_num >= 3
    show_margins = stage_num >= 4
    show_support_vectors = stage_num >= 4

    fig = build_decision_figure(
        X=X,
        y=y,
        model=main_model,
        title=(
            f"Alap SVM nézet – adathalmaz: {dataset_name} | "
            f"kernel: {kernel_label(kernel)} | C={C:.4f}"
        ),
        show_regions=show_regions,
        show_boundary=show_boundary,
        show_margins=show_margins,
        show_support_vectors=show_support_vectors,
        height=680,
    )

    left, right = st.columns([4, 1.55], gap="large")

    with left:
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Mit látsz most?")
        st.write(stage_explanation(stage, kernel))

        st.metric("Support vectorok", n_support)
        st.metric("Support vector arány", f"{100 * support_ratio:.1f}%")
        st.metric("Train accuracy", f"{100 * train_acc:.1f}%")

        st.markdown("---")
        st.subheader("Kernel röviden")
        st.write(short_kernel_explanation(kernel))

        if kernel == "linear" and w is not None:
            margin_width = 2.0 / np.linalg.norm(w)

            st.markdown("---")
            st.subheader("Lineáris modell")
            st.markdown(
                '<div class="formula">'
                f'{w[0]:.4f} · x₁ + {w[1]:.4f} · x₂ + {b:.4f} = 0'
                "</div>",
                unsafe_allow_html=True,
            )
            st.write(f"Margin szélesség ≈ {margin_width:.4f}")
        else:
            st.markdown("---")
            st.subheader("Nemlineáris modell")
            st.write(
                "Ebben az esetben a döntési határ általában nem egyetlen egyenes, "
                "hanem egy görbült elválasztó."
            )

    with st.expander("Elméleti háttér"):
        st.markdown(
            """
            - Az SVM a `f(x) = 0` döntési határt tanulja meg.
            - A `f(x) = +1` és `f(x) = -1` görbék/vonalak adják a margin két szélét.
            - A support vectorok azok a pontok, amelyek ezekhez a határokhoz a legközelebb vannak.
            - `linear` kernel esetén a határ egyenes.
            - `poly` és `rbf` kernelnél a határ görbült lehet.
            """
        )


# ============================================================
# TAB 2 – Kernel-trükk
# ============================================================
with tab2:
    st.subheader("A kernel-trükk lényege")

    st.markdown(
        """
        A kernel-trükk ötlete: az adatokat nem feltétlenül kell ténylegesen
        magasabb dimenzióba kirajzolni. Elég, ha egy olyan kernel-függvényt használunk,
        amely úgy viselkedik, mintha ott számolnánk belső szorzatot.

        A klasszikus példa a koncentrikus körök problémája:
        **2D-ben nem lineárisan szeparálható**, de az
        `φ(x₁, x₂) = (x₁, x₂, x₁² + x₂²)` leképezéssel **3D-ben már igen**.
        """
    )

    kleft, kright = st.columns(2, gap="large")
    fig2d, fig3d, threshold = build_explicit_kernel_trick_figures(
        n_samples=max(80, n_samples),
        noise=noise,
        seed=int(st.session_state.seed),
    )

    with kleft:
        st.plotly_chart(fig2d, use_container_width=True)

    with kright:
        st.plotly_chart(fig3d, use_container_width=True)

    st.markdown(
        f"""
        <div class="formula">
        φ(x₁, x₂) = (x₁, x₂, x₁² + x₂²)
        <br><br>
        2D-ben: x₁² + x₂² = konstans  → kör
        <br>
        3D-ben: z = konstans → sík
        <br><br>
        A mostani példában a szemléltetett küszöb kb. z = {threshold:.4f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Kernel-mátrix – hogyan „látja” egymást két minta?")

    km_col1, km_col2 = st.columns([1.2, 2.3], gap="large")

    with km_col1:
        kernel_matrix_choice = st.selectbox(
            "Kernel a mátrixhoz",
            ["linear", "poly", "rbf"],
            index=2,
            format_func=kernel_label,
            key="kernel_matrix_choice",
        )
        n_show = st.slider(
            "Megjelenített minták száma",
            min_value=20,
            max_value=min(90, len(X)),
            value=min(50, len(X)),
            step=5,
            key="n_show_kernel_matrix",
        )
        st.write(
            "A kernel-mátrix azt mutatja, hogy a modell szerint két pont mennyire "
            "hasonló egymáshoz a választott kernel alatt."
        )

    with km_col2:
        Kfig = build_kernel_matrix_figure(
            X=X,
            kernel_name=kernel_matrix_choice,
            gamma_value=gamma_value,
            degree=degree,
            coef0=coef0,
            n_show=n_show,
        )
        st.plotly_chart(Kfig, use_container_width=True)


# ============================================================
# TAB 3 – Kernel-összehasonlítás
# ============================================================
with tab3:
    st.subheader("Ugyanaz az adat, háromféle kernel")

    compare_cols = st.columns(3, gap="large")
    compare_kernels = ["linear", "poly", "rbf"]

    for col, kname in zip(compare_cols, compare_kernels):
        model_k = fit_svm(
            X=X,
            y=y,
            kernel=kname,
            C=C,
            gamma_value=gamma_value,
            degree=degree,
            coef0=coef0,
        )

        acc_k = accuracy_score(y, model_k.predict(X))
        nsv_k = len(model_k.named_steps["svc"].support_)

        fig_k = build_decision_figure(
            X=X,
            y=y,
            model=model_k,
            title=kernel_label(kname),
            show_regions=True,
            show_boundary=True,
            show_margins=True,
            show_support_vectors=True,
            height=430,
        )

        with col:
            st.plotly_chart(fig_k, use_container_width=True)
            st.metric("Train accuracy", f"{100 * acc_k:.1f}%")
            st.metric("Support vectorok", nsv_k)
            st.caption(short_kernel_explanation(kname))

    with st.expander("Mit figyelj meg?"):
        st.markdown(
            """
            - **Lineáris kernel**: egyenes határt tanul.
            - **Polinomiális kernel**: simább, algebrai jellegű görbületet tud modellezni.
            - **RBF kernel**: rugalmasabb, lokális mintázatokat is jól követ.
            - Nem minden adathalmazhoz ugyanaz a kernel a legjobb.
            """
        )


# ============================================================
# TAB 4 – Hiperaparaméterek hatása
# ============================================================
with tab4:
    st.subheader("A hiperaparaméterek szemléletes hatása")

    st.markdown(
        """
        Az SVM viselkedését főleg ezek szabályozzák:

        - **C**: mennyire büntesse a hibákat
        - **gamma**: RBF/polinomiális kernel esetén mennyire lokális legyen a döntési hatás
        """
    )

    st.markdown("### 1) `C` hatása ugyanazon a dataseten")

    c_low = 0.1
    c_mid = 1.0
    c_high = 20.0

    c_models = [
        ("Kicsi C = 0.1", fit_svm(X, y, kernel, c_low, gamma_value, degree, coef0)),
        ("Közepes C = 1.0", fit_svm(X, y, kernel, c_mid, gamma_value, degree, coef0)),
        ("Nagy C = 20.0", fit_svm(X, y, kernel, c_high, gamma_value, degree, coef0)),
    ]

    cols_c = st.columns(3, gap="large")
    for col, (label, model_c) in zip(cols_c, c_models):
        fig_c = build_decision_figure(
            X=X,
            y=y,
            model=model_c,
            title=label,
            show_regions=True,
            show_boundary=True,
            show_margins=True,
            show_support_vectors=True,
            height=420,
        )
        with col:
            st.plotly_chart(fig_c, use_container_width=True)
            acc_c = accuracy_score(y, model_c.predict(X))
            nsv_c = len(model_c.named_steps["svc"].support_)
            st.metric("Train accuracy", f"{100 * acc_c:.1f}%")
            st.metric("Support vectorok", nsv_c)

    st.markdown(
        """
        **Értelmezés**
        - kisebb `C` → nagyobb regularizáció, több hibát is elnéz, simább határ
        - nagyobb `C` → erősebben ráfeszül a pontokra, agresszívebb illesztés
        """
    )

    st.markdown("---")
    st.markdown("### 2) `gamma` hatása nemlineáris kerneleknél")

    if kernel in ["rbf", "poly"]:
        g_low = 0.1
        g_mid = 1.0
        g_high = 10.0

        g_models = [
            (f"Kicsi gamma = {g_low}", fit_svm(X, y, kernel, C, g_low, degree, coef0)),
            (f"Közepes gamma = {g_mid}", fit_svm(X, y, kernel, C, g_mid, degree, coef0)),
            (f"Nagy gamma = {g_high}", fit_svm(X, y, kernel, C, g_high, degree, coef0)),
        ]

        cols_g = st.columns(3, gap="large")
        for col, (label, model_g) in zip(cols_g, g_models):
            fig_g = build_decision_figure(
                X=X,
                y=y,
                model=model_g,
                title=label,
                show_regions=True,
                show_boundary=True,
                show_margins=True,
                show_support_vectors=True,
                height=420,
            )
            with col:
                st.plotly_chart(fig_g, use_container_width=True)
                acc_g = accuracy_score(y, model_g.predict(X))
                nsv_g = len(model_g.named_steps["svc"].support_)
                st.metric("Train accuracy", f"{100 * acc_g:.1f}%")
                st.metric("Support vectorok", nsv_g)

        st.markdown(
            """
            **Értelmezés**
            - kisebb `gamma` → simább, globálisabb döntési határ
            - nagyobb `gamma` → lokálisabb, kacskaringósabb döntési határ
            """
        )
    else:
        st.info("A `gamma` főleg `rbf` és `poly` kernelnél érdekes. Lineáris kernelnél lényegében nincs látványos szerepe.")


# ============================================================
# Lábléc
# ============================================================
st.markdown("---")
st.markdown(
    """
    **Javasolt kipróbálási sorrend**

    1. `Lineárisan szeparálható` + `linear`
    2. `Átfedő klaszterek` + `linear`, majd a `C` állítgatása
    3. `Két félhold` + `rbf`
    4. `Koncentrikus körök` + `Kernel-trükk` tab
    5. `XOR` + `poly` vagy `rbf`
    """
)