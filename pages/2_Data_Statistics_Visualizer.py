# pages/2_Data_Statistics_Visualizer.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import google.generativeai as genai
from google.generativeai import models as gen_models

# Helper to create PDF bytes for a short text
def text_to_pdf_bytes(text):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    y = h - 40
    for line in text.splitlines():
        if y < 40:
            c.showPage()
            y = h - 40
        c.drawString(40, y, line[:200])
        y -= 12
    c.save()
    buf.seek(0)
    return buf.getvalue()

st.set_page_config(page_title="Data Statistics Visualizer", layout="wide")
st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)
st.title("📉 Data Statistics Visualizer (Final Summary & Exports)")
st.caption("Advanced statistical views and the consolidated, LLM-polished summary + final exports. Uses the global dataset (uploaded once).")

# No uploader here — require global df
if 'df' not in st.session_state:
    st.warning("No global dataset found. Please upload your dataset on the 'Upload & Clean Data (Global)' page first.")
    st.stop()

df = st.session_state['df']

num_cols = df.select_dtypes(include=['number']).columns.tolist()
if not num_cols:
    st.warning("No numeric columns detected. Showing dataset head for inspection.")
    st.dataframe(df.head())
    st.stop()

st.subheader("📊 Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Mean (avg of numeric means)", f"{df[num_cols].mean().mean():.2f}")
col2.metric("Std Dev (avg)", f"{df[num_cols].std().mean():.2f}")
col3.metric("Median (avg)", f"{df[num_cols].median().mean():.2f}")

st.markdown("---")
st.subheader("🔬 Choose plot and columns")
plot_type = st.selectbox("Plot Type", ["Pair Matrix","Scatter Matrix","Box","Violin","KDE/Distribution","Heatmap (correlation)", "Top N Distribution"], key="stats_plot")
if plot_type in ["Box","Violin","KDE/Distribution","Top N Distribution"]:
    sel_col = st.selectbox("Choose numeric column", num_cols, key="stats_col")
if plot_type in ["Top N Distribution"]:
    top_n = st.slider("Top N categories (if categorical)", 3, 20, 8)

if plot_type == "Pair Matrix":
    st.write("Pair matrix (interactive)")
    fig = px.scatter_matrix(df[num_cols], dimensions=num_cols[:6], title="Scatter matrix (first 6 numeric cols)")
    st.plotly_chart(fig, use_container_width=True)
elif plot_type == "Scatter Matrix":
    st.write("Scatter matrix (Plotly) - try subset of numeric columns")
    cols = st.multiselect("Pick columns (2-6)", num_cols, default=num_cols[:4])
    if len(cols) >= 2:
        fig = px.scatter_matrix(df[cols], dimensions=cols, title="Scatter Matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pick at least 2 columns.")
elif plot_type == "Box":
    fig = px.box(df, y=sel_col, points="outliers", title=f"Boxplot: {sel_col}")
    st.plotly_chart(fig, use_container_width=True)
elif plot_type == "Violin":
    fig = px.violin(df, y=sel_col, box=True, title=f"Violin plot: {sel_col}")
    st.plotly_chart(fig, use_container_width=True)
elif plot_type == "KDE/Distribution":
    fig = px.histogram(df, x=sel_col, nbins=30, marginal="box", title=f"Distribution + Box: {sel_col}")
    st.plotly_chart(fig, use_container_width=True)
elif plot_type == "Heatmap (correlation)":
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)
elif plot_type == "Top N Distribution":
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if cat_cols:
        sel_cat = st.selectbox("Choose categorical column", cat_cols)
        counts = df[sel_cat].value_counts().head(top_n).reset_index()
        counts.columns = [sel_cat, "count"]
        fig = px.bar(counts, x=sel_cat, y="count", title=f"Top {top_n} {sel_cat}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns to show counts.")

st.markdown("---")

# Consolidated LLM-polished summary section (final page)
st.header("📝 Consolidated LLM-polished Summary (Final)")
st.markdown("This is the single place to generate an LLM-polished summary for the entire dataset and the auto-analysis results. It uses the configured Gemini model (select model in main page).")

# Small helper functions (re-implemented local to this file to avoid cross-dependency)
def top_correlations_local(df, n=6):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return []
    corr = num.corr().abs()
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    top = pairs.head(n)
    result = []
    for (a,b), val in top.items():
        signed = num[a].corr(num[b])
        result.append({"x": a, "y": b, "corr_abs": val, "corr": signed})
    return result

def concise_rule_insights_local(df, top_corrs, outlier_counts):
    insights = []
    insights.append(f"The dataset has {df.shape[0]} records and {df.shape[1]} columns.")
    num_cols_local = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_local = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    insights.append(f"{len(num_cols_local)} numerical columns, {len(cat_cols_local)} categorical/text columns.")
    if top_corrs:
        for t in top_corrs:
            s = t['corr']
            direction = "positive" if s > 0 else "negative"
            insights.append(f"Column '{t['x']}' and '{t['y']}' have a strong {direction} correlation (r = {s:.2f}).")
    else:
        insights.append("No strong numeric correlations detected.")
    if outlier_counts:
        high_outliers = {k:v for k,v in outlier_counts.items() if v > 0}
        if high_outliers:
            top = sorted(high_outliers.items(), key=lambda x: x[1], reverse=True)[:3]
            for col, cnt in top:
                insights.append(f"Column '{col}' has {cnt} potential outliers (IQR or z-score based).")
        else:
            insights.append("No significant outlier counts detected with current thresholds.")
    return insights

def detect_outliers_iqr_local(df):
    num = df.select_dtypes(include=[np.number])
    outlier_counts = {}
    for col in num.columns:
        q1 = num[col].quantile(0.25)
        q3 = num[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            outlier_counts[col] = 0
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        mask = (num[col] < low) | (num[col] > high)
        outlier_counts[col] = int(mask.sum())
    return outlier_counts

# Controls to run auto-analysis here (final)
if st.button("Run final auto-analysis & prepare LLM summary"):
    with st.spinner("Running final analysis..."):
        top_corrs = top_correlations_local(df, n=6)
        out_counts = detect_outliers_iqr_local(df)
        rule_insights = concise_rule_insights_local(df, top_corrs, out_counts)
        st.subheader("Auto Analysis — concise findings")
        for i, ins in enumerate(rule_insights, start=1):
            st.write(f"{i}. {ins}")
        # allow LLM polishing
        # get model config from session if available, else default
        model_name = st.session_state.get('selected_model', "gemini-flash-latest")
        # But the main app already has a model selectbox; if user selected there, ensure they saved it to session_state['selected_model']
        if 'selected_model' in st.session_state:
            model_name = st.session_state['selected_model']
        st.info(f"Using model: {model_name}")
        # build prompt
        df_head_csv = df.head(6).to_csv(index=False)
        findings_text = "\n".join(rule_insights)
        prompt = (
            "You are an assistant that converts bullet analytical findings into a clear, concise, "
            "non-technical report for users who are not data scientists.\n\n"
            f"DATA SAMPLE (first rows CSV):\n{df_head_csv}\n\n"
            f"FINDINGS (bullet points):\n{findings_text}\n\n"
            "Write a friendly, 6-10 sentence summary highlighting the most important insights, "
            "what they mean, and a couple of practical suggestions the user might act on. "
            "Avoid hallucination: use only the facts given."
        )
        try:
            # configure genai if not already configured
            try:
                genai_key = os.getenv("GENAI_API_KEY")
                if genai_key:
                    genai.configure(api_key=genai_key)
            except Exception:
                pass
            model = genai.GenerativeModel(model_name=model_name)
            resp = model.generate_content([prompt])
            polished = resp.text if hasattr(resp, "text") else str(resp)
        except Exception as e:
            polished = f"(LLM summary failed: {e})\n\nRaw findings:\n{findings_text}"

        st.subheader("Polished Summary (LLM)")
        st.write(polished)
        st.download_button("Download Polished TXT", polished, "polished_summary.txt")
        st.download_button("Download Polished PDF", text_to_pdf_bytes(polished), "polished_summary.pdf", mime="application/pdf")

# Also provide raw export of the dataset and the analysis
# Also provide raw export of the dataset and the analysis
st.markdown("---")
st.header("Exports")

# --- Helper to export Excel properly ---
import io
from openpyxl import Workbook

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Dataset")
excel_buffer.seek(0)

st.download_button(
    label="Download dataset (Excel)",
    data=excel_buffer,
    file_name="dataset.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download dataset (CSV)",
    data=csv_bytes,
    file_name="dataset.csv",
    mime="text/csv"
)

st.info("You can also use the Auto Analysis button above to generate a final LLM-polished summary and PDFs.")
