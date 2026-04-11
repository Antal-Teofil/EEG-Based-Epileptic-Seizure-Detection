import streamlit as st


def configure_page():
    st.set_page_config(
        page_title="SVM Explorer – Interaktív vizualizáció",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_global_styles():
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


def render_hero():
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


def render_info_cards():
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
                    A döntési határ és a legközelebbi pontok közti biztonsági sáv.
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


def render_pills(dataset_name: str, kernel_name: str, C: float, gamma_value: float, n_points: int, acc: float):
    st.markdown(
        f"""
        <div class="pillrow">
            <div class="pill">Adathalmaz: {dataset_name}</div>
            <div class="pill">Kernel: {kernel_name}</div>
            <div class="pill">C = {C:.4f}</div>
            <div class="pill">gamma = {gamma_value:.4f}</div>
            <div class="pill">Pontok száma = {n_points}</div>
            <div class="pill">Train accuracy = {100 * acc:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )