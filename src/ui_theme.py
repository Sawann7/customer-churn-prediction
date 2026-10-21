"""
Shared UI Theme & Design Components for the Streamlit Application.
Provides sleek CSS cards, modern typography, responsive metric pills,
dual-tab separation, and a rock-solid Dark/Light Mode switcher.
"""

import streamlit as st


def get_current_theme() -> str:
    """Retrieves current theme from session state, defaulting to 'light'."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    return st.session_state["theme"]


def render_theme_toggle() -> str:
    """Renders prominent side-by-side buttons for switching between Light and Dark mode."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"

    curr = st.session_state["theme"]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**🌓 Display Theme:** `{'🌙 Dark Mode' if curr == 'dark' else '☀️ Light Mode'}`")
    c1, c2 = st.sidebar.columns(2)

    with c1:
        if c1.button("☀️ Light", key="btn_theme_light", use_container_width=True, type="primary" if curr == "light" else "secondary"):
            if curr != "light":
                st.session_state["theme"] = "light"
                st.rerun()

    with c2:
        if c2.button("🌙 Dark", key="btn_theme_dark", use_container_width=True, type="primary" if curr == "dark" else "secondary"):
            if curr != "dark":
                st.session_state["theme"] = "dark"
                st.rerun()

    return st.session_state["theme"]


def apply_custom_theme(theme: str = None):
    """Applies high-specificity modern CSS styling across all Streamlit pages according to active theme."""
    if theme is None:
        theme = get_current_theme()

    if theme == "dark":
        # Comprehensive Dark Theme Palette
        css_rules = """
        /* Main background and app containers */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background-color: #0b0f19 !important;
            color: #f8fafc !important;
        }

        /* Sidebar in Dark Mode */
        section[data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background-color: #111827 !important;
            color: #f8fafc !important;
            border-right: 1px solid #1f2937 !important;
        }

        /* Top Header */
        [data-testid="stHeader"] {
            background-color: rgba(11, 15, 25, 0.95) !important;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc !important;
        }
        p, label, .stMarkdown, .stText, span {
            color: #cbd5e1 !important;
        }

        /* Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #1e40af 100%) !important;
            border-radius: 12px;
            padding: 1.6rem 2rem;
            color: #ffffff;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
            border: 1px solid #1e3a8a;
        }
        .hero-title {
            color: #ffffff !important;
        }
        .hero-subtitle {
            color: #93c5fd !important;
        }

        /* Glassmorphic Metric Cards */
        .glass-card {
            background: #161f30 !important;
            border: 1px solid #243247 !important;
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }
        .card-number {
            color: #ffffff !important;
        }
        .card-label {
            color: #94a3b8 !important;
        }
        .delta-positive { color: #34d399 !important; }
        .delta-negative { color: #f87171 !important; }

        /* Form Inputs & Selectboxes */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        input {
            background-color: #1e293b !important;
            color: #ffffff !important;
            border-color: #334155 !important;
        }
        div[data-baseweb="popover"], div[data-baseweb="menu"] {
            background-color: #1e293b !important;
            color: #ffffff !important;
        }
        li[data-baseweb="menu-item"] {
            color: #f8fafc !important;
        }

        /* Dataframe */
        [data-testid="stDataFrame"] {
            background-color: #161f30 !important;
        }

        /* Status Pills */
        .pill-success {
            background-color: rgba(16, 185, 129, 0.2) !important;
            color: #34d399 !important;
            border: 1px solid #059669 !important;
        }
        .pill-primary {
            background-color: rgba(59, 130, 246, 0.2) !important;
            color: #60a5fa !important;
            border: 1px solid #2563eb !important;
        }
        .pill-warning {
            background-color: rgba(245, 158, 11, 0.2) !important;
            color: #fbbf24 !important;
            border: 1px solid #d97706 !important;
        }

        /* Documentation & Explanation Box */
        .doc-box {
            background-color: #111827 !important;
            border-left: 4px solid #3b82f6 !important;
            border-top: 1px solid #1f2937 !important;
            border-right: 1px solid #1f2937 !important;
            border-bottom: 1px solid #1f2937 !important;
        }
        .doc-title {
            color: #93c5fd !important;
        }
        .doc-text {
            color: #cbd5e1 !important;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 2px solid #1e293b !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #94a3b8 !important;
        }
        .stTabs [aria-selected="true"] {
            color: #60a5fa !important;
            border-bottom: 3px solid #60a5fa !important;
        }
        """
    else:
        # Comprehensive Light Theme Palette
        css_rules = """
        /* Main background and app containers */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background-color: #f8fafc !important;
            color: #0f172a !important;
        }

        /* Sidebar in Light Mode */
        section[data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border-right: 1px solid #e2e8f0 !important;
        }

        /* Top Header */
        [data-testid="stHeader"] {
            background-color: rgba(248, 250, 252, 0.95) !important;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #0f172a !important;
        }
        p, label, .stMarkdown, .stText, span {
            color: #334155 !important;
        }

        /* Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 60%, #0284c7 100%) !important;
            border-radius: 12px;
            padding: 1.6rem 2rem;
            color: #ffffff;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.18);
        }
        .hero-title {
            color: #ffffff !important;
        }
        .hero-subtitle {
            color: #e0f2fe !important;
        }

        /* Glassmorphic Metric Cards */
        .glass-card {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .card-number {
            color: #0f172a !important;
        }
        .card-label {
            color: #64748b !important;
        }
        .delta-positive { color: #16a34a !important; }
        .delta-negative { color: #dc2626 !important; }

        /* Form Inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        input {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border-color: #cbd5e1 !important;
        }

        /* Status Pills */
        .pill-success {
            background-color: #dcfce7 !important;
            color: #15803d !important;
            border: 1px solid #bbf7d0 !important;
        }
        .pill-primary {
            background-color: #eff6ff !important;
            color: #1d4ed8 !important;
            border: 1px solid #bfdbfe !important;
        }
        .pill-warning {
            background-color: #fef9c3 !important;
            color: #854d0e !important;
            border: 1px solid #fef08a !important;
        }

        /* Documentation Box */
        .doc-box {
            background-color: #ffffff !important;
            border-left: 4px solid #2563eb !important;
            border-top: 1px solid #e2e8f0 !important;
            border-right: 1px solid #e2e8f0 !important;
            border-bottom: 1px solid #e2e8f0 !important;
        }
        .doc-title {
            color: #1e3a8a !important;
        }
        .doc-text {
            color: #334155 !important;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 2px solid #e2e8f0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #64748b !important;
        }
        .stTabs [aria-selected="true"] {
            color: #2563eb !important;
            border-bottom: 3px solid #2563eb !important;
        }
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        {css_rules}

        .pill {{
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }}

        .stButton>button {{
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 12px;
            padding-bottom: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 44px;
            white-space: pre-wrap;
            border-radius: 6px 6px 0 0;
            padding: 8px 18px;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, badge_text: str = "Production Machine Learning"):
    """Renders a modern, visually appealing gradient hero banner."""
    st.markdown(
        f"""
        <div class="hero-banner">
            <span class="pill pill-primary" style="background-color: rgba(255,255,255,0.18) !important; color: #ffffff !important; border-color: rgba(255,255,255,0.3) !important; margin-bottom: 0.6rem;">{badge_text}</span>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
