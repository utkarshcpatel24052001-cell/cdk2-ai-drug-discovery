st.markdown(
    """
<style>
/* 1) Base page font */
html, body {
    font-family: "Times New Roman", Times, serif !important;
    font-size: 18px !important;
}

/* 2) Only style typical content areas (avoid global div/span override) */
.stMarkdown, .stMarkdown p, .stMarkdown li {
    font-family: "Times New Roman", Times, serif !important;
    font-size: 18px !important;
    line-height: 1.35 !important;
}

/* 3) Metrics */
div[data-testid="stMetricLabel"] p {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #1f77b4 !important;
    line-height: 1.2 !important;
}
div[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 500 !important;
    line-height: 1.1 !important;
}

/* 4) Buttons */
.stButton>button {
    height: 3em;
    font-size: 18px !important;
    border-radius: 6px;
}

/* 5) Expander header: force clean layout */
div[data-testid="stExpander"] summary {
    font-size: 18px !important;
    line-height: 1.2 !important;
}
</style>
""",
    unsafe_allow_html=True,
)
