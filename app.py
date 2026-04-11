import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score

# python3 -m streamlit run app.py

from svm_explorer.constants import DATASETS, KERNELS, STAGES
from svm_explorer.data import make_dataset
from svm_explorer.explanations import (
    kernel_label,
    short_kernel_explanation,
    stage_explanation,
)
from svm_explorer.modeling import (
    fit_svm,
    get_linear_equation_in_original_space,
)
from svm_explorer.plots import (
    build_decision_figure,
    build_explicit_kernel_trick_figures,
    build_kernel_matrix_figure,
)
from svm_explorer.styles import (
    apply_global_styles,
    configure_page,
    render_hero,
    render_info_cards,
    render_pills,
)


def init_state():
    if "seed" not in st.session_state:
        st.session_state.seed = 42


def render_sidebar():
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

    return {
        "dataset_name": dataset_name,
        "kernel": kernel,
        "C": C,
        "gamma_value": gamma_value,
        "degree": degree,
        "coef0": coef0,
        "n_samples": n_samples,
        "noise": noise,
        "seed": int(st.session_state.seed),
    }


def build_main_state(settings):
    X, y = make_dataset(
        dataset_name=settings["dataset_name"],
        n_samples=settings["n_samples"],
        noise=settings["noise"],
        seed=settings["seed"],
    )

    model = fit_svm(
        X=X,
        y=y,
        kernel=settings["kernel"],
        C=settings["C"],
        gamma_value=settings["gamma_value"],
        degree=settings["degree"],
        coef0=settings["coef0"],
    )

    y_pred = model.predict(X)
    train_acc = accuracy_score(y, y_pred)

    svc = model.named_steps["svc"]
    n_support = int(len(svc.support_))
    support_ratio = n_support / len(X)

    w, b = get_linear_equation_in_original_space(model)

    return {
        "X": X,
        "y": y,
        "model": model,
        "train_acc": train_acc,
        "n_support": n_support,
        "support_ratio": support_ratio,
        "w": w,
        "b": b,
    }


def render_basic_tab(settings, state):
    stage = st.select_slider("Megjelenítési lépés", options=STAGES, value=STAGES[3])

    stage_num = STAGES.index(stage) + 1
    show_regions = stage_num >= 2
    show_boundary = stage_num >= 3
    show_margins = stage_num >= 4
    show_support_vectors = stage_num >= 4

    fig = build_decision_figure(
        X=state["X"],
        y=state["y"],
        model=state["model"],
        title=(
            f"Alap SVM nézet – adathalmaz: {settings['dataset_name']} | "
            f"kernel: {kernel_label(settings['kernel'])} | C={settings['C']:.4f}"
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
        st.write(stage_explanation(stage, settings["kernel"]))

        st.metric("Support vectorok", state["n_support"])
        st.metric("Support vector arány", f"{100 * state['support_ratio']:.1f}%")
        st.metric("Train accuracy", f"{100 * state['train_acc']:.1f}%")

        st.markdown("---")
        st.subheader("Kernel röviden")
        st.write(short_kernel_explanation(settings["kernel"]))

        if settings["kernel"] == "linear" and state["w"] is not None:
            margin_width = 2.0 / np.linalg.norm(state["w"])

            st.markdown("---")
            st.subheader("Lineáris modell")
            st.markdown(
                '<div class="formula">'
                f'{state["w"][0]:.4f} · x₁ + {state["w"][1]:.4f} · x₂ + {state["b"]:.4f} = 0'
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
            - Az SVM az `f(x)=0` döntési határt tanulja meg.
            - Az `f(x)=+1` és `f(x)=-1` görbék vagy vonalak adják a margin két szélét.
            - A support vectorok azok a pontok, amelyek ezekhez a határokhoz a legközelebb vannak.
            - `linear` kernel esetén a határ egyenes.
            - `poly` és `rbf` kernelnél a határ görbült lehet.
            """
        )


def render_kernel_trick_tab(settings, state):
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
        n_samples=max(80, settings["n_samples"]),
        noise=settings["noise"],
        seed=settings["seed"],
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
        2D-ben: x₁² + x₂² = konstans → kör
        <br>
        3D-ben: z = konstans → sík
        <br><br>
        A mostani példában a szemléltetett küszöb kb. z = {threshold:.4f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Kernel-mátrix – hogyan látja egymást két minta?")

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
            max_value=min(90, len(state["X"])),
            value=min(50, len(state["X"])),
            step=5,
            key="n_show_kernel_matrix",
        )
        st.write(
            "A kernel-mátrix azt mutatja, hogy a modell szerint két pont "
            "mennyire hasonló egymáshoz a választott kernel alatt."
        )

    with km_col2:
        kfig = build_kernel_matrix_figure(
            X=state["X"],
            kernel_name=kernel_matrix_choice,
            gamma_value=settings["gamma_value"],
            degree=settings["degree"],
            coef0=settings["coef0"],
            n_show=n_show,
        )
        st.plotly_chart(kfig, use_container_width=True)


def render_kernel_compare_tab(settings, state):
    st.subheader("Ugyanaz az adat, háromféle kernel")

    compare_cols = st.columns(3, gap="large")
    compare_kernels = ["linear", "poly", "rbf"]

    for col, kname in zip(compare_cols, compare_kernels):
        model_k = fit_svm(
            X=state["X"],
            y=state["y"],
            kernel=kname,
            C=settings["C"],
            gamma_value=settings["gamma_value"],
            degree=settings["degree"],
            coef0=settings["coef0"],
        )

        acc_k = accuracy_score(state["y"], model_k.predict(state["X"]))
        nsv_k = len(model_k.named_steps["svc"].support_)

        fig_k = build_decision_figure(
            X=state["X"],
            y=state["y"],
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


def render_hyperparameter_tab(settings, state):
    st.subheader("A hiperaparaméterek szemléletes hatása")

    st.markdown(
        """
        Az SVM viselkedését főleg ezek szabályozzák:

        - **C**: mennyire büntesse a hibákat
        - **gamma**: RBF vagy polinomiális kernel esetén mennyire lokális legyen a döntési hatás
        """
    )

    st.markdown("### 1) `C` hatása ugyanazon a dataseten")

    c_models = [
        ("Kicsi C = 0.1", fit_svm(state["X"], state["y"], settings["kernel"], 0.1, settings["gamma_value"], settings["degree"], settings["coef0"])),
        ("Közepes C = 1.0", fit_svm(state["X"], state["y"], settings["kernel"], 1.0, settings["gamma_value"], settings["degree"], settings["coef0"])),
        ("Nagy C = 20.0", fit_svm(state["X"], state["y"], settings["kernel"], 20.0, settings["gamma_value"], settings["degree"], settings["coef0"])),
    ]

    cols_c = st.columns(3, gap="large")
    for col, (label, model_c) in zip(cols_c, c_models):
        fig_c = build_decision_figure(
            X=state["X"],
            y=state["y"],
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
            acc_c = accuracy_score(state["y"], model_c.predict(state["X"]))
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

    if settings["kernel"] in ["rbf", "poly"]:
        g_models = [
            (f"Kicsi gamma = 0.1", fit_svm(state["X"], state["y"], settings["kernel"], settings["C"], 0.1, settings["degree"], settings["coef0"])),
            (f"Közepes gamma = 1.0", fit_svm(state["X"], state["y"], settings["kernel"], settings["C"], 1.0, settings["degree"], settings["coef0"])),
            (f"Nagy gamma = 10.0", fit_svm(state["X"], state["y"], settings["kernel"], settings["C"], 10.0, settings["degree"], settings["coef0"])),
        ]

        cols_g = st.columns(3, gap="large")
        for col, (label, model_g) in zip(cols_g, g_models):
            fig_g = build_decision_figure(
                X=state["X"],
                y=state["y"],
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
                acc_g = accuracy_score(state["y"], model_g.predict(state["X"]))
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
        st.info(
            "A `gamma` főleg `rbf` és `poly` kernelnél érdekes. "
            "Lineáris kernelnél nincs látványos szerepe."
        )


def main():
    configure_page()
    apply_global_styles()
    init_state()

    render_hero()
    settings = render_sidebar()
    state = build_main_state(settings)

    render_info_cards()
    render_pills(
        dataset_name=settings["dataset_name"],
        kernel_name=kernel_label(settings["kernel"]),
        C=settings["C"],
        gamma_value=settings["gamma_value"],
        n_points=len(state["X"]),
        acc=state["train_acc"],
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Alapötlet",
            "Kernel-trükk",
            "Kernel-összehasonlítás",
            "Hiperaparaméterek hatása",
        ]
    )

    with tab1:
        render_basic_tab(settings, state)

    with tab2:
        render_kernel_trick_tab(settings, state)

    with tab3:
        render_kernel_compare_tab(settings, state)

    with tab4:
        render_hyperparameter_tab(settings, state)

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


if __name__ == "__main__":
    main()