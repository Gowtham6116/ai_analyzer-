# streamlit_app.py

import os
import io
import math
import pandas as pd
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import models as gen_models
import json

# ====== CONFIG ======
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_MODEL = "gemini-flash-latest"
ENV_API_KEY = "GENAI_API_KEY"
MAX_PROMPT_CHARS = 120000

# Load env
load_dotenv()

# ----------------- Helper Functions -----------------
def init_genai():
    api_key = os.getenv(ENV_API_KEY)
    if not api_key:
        st.error(f"❌ Environment variable {ENV_API_KEY} not set. Please set it in .env.")
        st.stop()
    genai.configure(api_key=api_key)

def list_available_models():
    """Get list of available models via API."""
    try:
        resp = gen_models.list_models()
        names = [m.name for m in resp.models]
        return names
    except Exception as e:
        st.warning(f"Could not list models: {e}")
        return []

def ask_model(prompt: str, model_name: str):
    """Call the model via generate_content."""
    model = genai.GenerativeModel(model_name=model_name)
    resp = model.generate_content([prompt])
    return resp.text if hasattr(resp, "text") else str(resp)

def chunk_text(text, max_chars=MAX_PROMPT_CHARS):
    if len(text) <= max_chars:
        return [text]
    chunks = []
    idx = 0
    while idx < len(text):
        chunk = text[idx: idx + max_chars]
        last_nl = chunk.rfind("\n")
        if last_nl > int(max_chars * 0.6):
            chunk = chunk[:last_nl]
            idx += last_nl
        else:
            idx += max_chars
        chunks.append(chunk)
    return chunks

def read_pdf(bts):
    return "\n".join([p.extract_text() or "" for p in PdfReader(io.BytesIO(bts)).pages])

def read_docx(bts):
    return "\n".join([p.text for p in Document(io.BytesIO(bts)).paragraphs])

def read_excel_to_df(bts):
    return pd.read_excel(io.BytesIO(bts), sheet_name=0)

def read_csv_to_df(bts):
    return pd.read_csv(io.BytesIO(bts))

def df_to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    buf.seek(0)
    return buf.getvalue()

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

# ----------------- Analysis Helpers -----------------
def top_correlations(df: pd.DataFrame, n=5):
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

def correlation_heatmap_fig(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return None
    corr = num.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix")
    return fig

def detect_outliers_iqr(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    outlier_counts = {}
    outlier_indices = {}
    for col in num.columns:
        q1 = num[col].quantile(0.25)
        q3 = num[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            outlier_counts[col] = 0
            outlier_indices[col] = []
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        mask = (num[col] < low) | (num[col] > high)
        outlier_counts[col] = int(mask.sum())
        outlier_indices[col] = num[mask].index.tolist()
    return outlier_counts, outlier_indices

def detect_outliers_zscore(df: pd.DataFrame, z_thresh=3.0):
    num = df.select_dtypes(include=[np.number])
    outlier_counts = {}
    outlier_indices = {}
    for col in num.columns:
        colvals = num[col].dropna()
        mean = colvals.mean()
        std = colvals.std(ddof=0)
        if std == 0 or pd.isna(std):
            outlier_counts[col] = 0
            outlier_indices[col] = []
            continue
        z = ((num[col] - mean) / std).abs()
        mask = z > z_thresh
        outlier_counts[col] = int(mask.sum())
        outlier_indices[col] = num[mask].index.tolist()
    return outlier_counts, outlier_indices

def concise_rule_insights(df: pd.DataFrame, top_corrs, outlier_counts):
    insights = []
    # Basic dataset info
    insights.append(f"The dataset has {df.shape[0]} records and {df.shape[1]} columns.")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    insights.append(f"{len(num_cols)} numerical columns, {len(cat_cols)} categorical/text columns.")
    # Top correlations
    if top_corrs:
        for t in top_corrs:
            s = t['corr']
            direction = "positive" if s > 0 else "negative"
            insights.append(f"Column '{t['x']}' and '{t['y']}' have a strong {direction} correlation (r = {s:.2f}).")
    else:
        insights.append("No strong numeric correlations detected.")
    # Outliers
    if outlier_counts:
        high_outliers = {k:v for k,v in outlier_counts.items() if v > 0}
        if high_outliers:
            top = sorted(high_outliers.items(), key=lambda x: x[1], reverse=True)[:3]
            for col, cnt in top:
                insights.append(f"Column '{col}' has {cnt} potential outliers (IQR or z-score based).")
        else:
            insights.append("No significant outlier counts detected with current thresholds.")
    return insights

def generate_polished_report_with_llm(findings_text, df_head_csv, model_name):
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
        return ask_model(prompt, model_name)
    except Exception as e:
        return f"(LLM summary failed: {e})\n\nRaw findings:\n{findings_text}"

def safe_parse_json_from_model(s):
    """Try to extract JSON safely from a model response. Returns dict or raises."""
    if not s:
        raise ValueError("Empty response")
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            pass
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        pass
    raise ValueError("Could not parse JSON from model response")

def safe_scatter_plot(df, x, y, title=None):
    try:
        fig = px.scatter(df, x=x, y=y, trendline="ols", title=title)
        return fig
    except Exception:
        try:
            fig = px.scatter(df, x=x, y=y, title=title)
            return fig
        except Exception:
            return None

# --- Streamlit UI ---
st.set_page_config(page_title="AI Analyzer", layout="wide")
st.title("AI Analyzer — with dynamic model fallback and pattern finder")

# Initialize API
init_genai()

# Option to pick model
available = list_available_models()
model_name = st.selectbox("Select model", options=[DEFAULT_MODEL] + available, index=0)

# Sidebar settings
st.sidebar.header("Settings")
max_mb = st.sidebar.number_input("Max upload size (MB)", value=int(DEFAULT_MAX_BYTES / (1024 * 1024)), min_value=1, max_value=200)
MAX_UPLOAD_BYTES = int(max_mb * 1024 * 1024)

st.sidebar.subheader("Outlier Detection Settings")
z_threshold = st.sidebar.slider("Z-score threshold (for z-method)", min_value=2.0, max_value=6.0, value=3.0, step=0.5)
use_method = st.sidebar.selectbox("Outlier method", options=["IQR (default)","Z-score"])

# NOTE: No uploader on this page. Expect dataset to be loaded via Global Upload page.
if 'df' not in st.session_state:
    st.warning("No global dataset loaded. Please go to the 'Upload & Clean Data (Global)' page and upload your CSV/Excel there.")
    st.info("After uploading on the Global page, return here. Pages share the dataset via session state.")
    # Helpful pointer (users can use the sidebar pages or folder-based navigation in Streamlit)
    st.markdown("➡️ Go to: **Upload & Clean Data (Global)** (in the Pages sidebar) and upload once.")
    st.stop()

df = st.session_state['df']

# Show small header with file name & basic shape (no big preview)
st.markdown(f"**Dataset:** {st.session_state.get('uploaded_file_name','(global upload)')} — Rows: {df.shape[0]}, Cols: {df.shape[1]}")

st.markdown("---")
st.header("Ask AI (data-aware)")

q = st.text_input("Ask a question about your dataset")
if st.button("Ask"):
    with st.spinner("Calling model..."):
        # build context from columns + sample rows
        context = f"Columns: {', '.join(df.columns)}\n\nTop rows:\n{df.head(50).to_csv(index=False)}"
        prompt = f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{q}"
        try:
            ans = ask_model(prompt, model_name)
            st.subheader("Answer")
            st.write(ans)
            st.download_button("Download TXT", ans, "answer.txt")
            st.download_button("Download PDF", text_to_pdf_bytes(ans), "answer.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Failed generate: {e}")

st.markdown("---")
st.header("📊 Manual Graph Creator (Bar, Line, Pie, Histogram)")

num_cols = df.select_dtypes(include="number").columns.tolist()
cols = df.columns.tolist()

if cols:
    x = st.selectbox("X-axis (for pie, use categorical column)", [None] + cols, index=0)
    y = st.selectbox("Y-axis (numeric or numeric-like)", cols, index=0)
    t = st.selectbox("Chart Type", ["line","bar","pie","histogram"])
    if st.button("Generate Chart"):
        try:
            plot_df = df.copy()
            # Robust pie handling: if pie selected, try to aggregate numeric y by categorical x
            if t == "pie":
                if x is None:
                    st.error("Pie chart requires a categorical 'X-axis' to group by.")
                else:
                    # try convert y to numeric
                    plot_df[y] = pd.to_numeric(plot_df[y].astype(str).str.replace(',', ''), errors='coerce')
                    grouped = plot_df.groupby(x)[y].sum().reset_index()
                    if grouped[y].isna().all():
                        st.error("Could not interpret the Y column as numeric for pie chart aggregation.")
                    else:
                        fig = px.pie(grouped, names=x, values=y, title=f"{y} distribution by {x}")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                if t == "line":
                    fig = px.line(plot_df, x=x, y=y, title=f"{t.capitalize()} Chart: {y} vs {x}")
                elif t == "bar":
                    fig = px.bar(plot_df, x=x, y=y, title=f"{t.capitalize()} Chart: {y} vs {x}")
                else:
                    fig = px.histogram(plot_df, x=y, title=f"{t.capitalize()} Chart: {y}")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart failed: {e}")

st.markdown("---")
st.header("📊 Smart AI Chart Generator")
st.caption("Automatically creates the most relevant chart based on your request and dataset (supports pie charts).")
user_query = st.text_input("Ask AI to generate a chart (e.g., 'Show total base price per player')", key="ai_graph_query")
if st.button("Generate AI Chart"):
    with st.spinner("Analyzing your query..."):
        columns = ", ".join(df.columns)
        sample = df.head(20).to_csv(index=False)
        prompt = f"""
You are an assistant that selects best columns for a chart from a dataset.
Columns: {columns}
Data sample:
{sample}

User request: {user_query}

Respond as JSON exactly like:
{{"x": "column_for_x_axis", "y": "column_for_y_axis", "chart": "bar|line|pie|histogram"}}
Only return JSON.
"""
        try:
            suggestion = ask_model(prompt, model_name)
            try:
                info = safe_parse_json_from_model(suggestion)
            except Exception:
                # fallback heuristics
                info = {}
                for col in df.columns:
                    if col in suggestion and "x" not in info:
                        info["x"] = col
                    elif col in suggestion and "x" in info and "y" not in info:
                        info["y"] = col
                info.setdefault("chart", "bar")
                info.setdefault("y", df.select_dtypes(include="number").columns.tolist()[0] if len(df.select_dtypes(include="number").columns)>0 else df.columns[0])
                info.setdefault("x", info.get("x", df.columns[0]))
            x_col, y_col, chart = info.get("x"), info.get("y"), info.get("chart")
            if x_col not in df.columns or y_col not in df.columns:
                st.warning("⚠️ AI suggested columns are not present in this dataset. Showing column list to pick manually.")
                st.write("Columns:", df.columns.tolist())
            else:
                st.success(f"✅ AI chose {chart} chart with X='{x_col}' and Y='{y_col}'")
                plot_df = df[[x_col, y_col]].dropna().copy()
                # robust pie: aggregate if needed
                try:
                    if chart == "bar":
                        fig = px.bar(plot_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    elif chart == "line":
                        fig = px.line(plot_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    elif chart == "pie":
                        plot_df[y_col] = pd.to_numeric(plot_df[y_col].astype(str).str.replace(',', ''), errors='coerce')
                        grouped = plot_df.groupby(x_col)[y_col].sum().reset_index()
                        fig = px.pie(grouped, names=x_col, values=y_col, title=f"{y_col} distribution by {x_col}")
                    else:
                        fig = px.histogram(plot_df, x=y_col, title=f"Distribution of {y_col}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Plotting failed: {e}")
        except Exception as e:
            st.error(f"Could not generate AI chart: {e}")

# ----------------- Auto Pattern & Anomaly Finder -----------------
st.markdown("---")
st.header("🧠 Auto Pattern & Anomaly Finder")
st.caption("Automatically finds top correlations, anomalies, and produces plain-language insights.")

if st.button("Run Auto Analysis"):
    with st.spinner("Analyzing dataset..."):
        top_corrs = top_correlations(df, n=6)
        if top_corrs:
            st.subheader("💫 Strong Relationships Between Columns")
            corr_df = pd.DataFrame(top_corrs)
            corr_df_display = corr_df.copy()
            corr_df_display["corr_abs"] = corr_df_display["corr_abs"].round(3)
            corr_df_display["corr"] = corr_df_display["corr"].round(3)
            st.table(corr_df_display[['x','y','corr']].rename(columns={"corr":"r (signed)"}))
            top_pair = top_corrs[0]
            try:
                sc_fig = safe_scatter_plot(df, top_pair['x'], top_pair['y'],
                                          title=f"Scatter: {top_pair['x']} vs {top_pair['y']} (r={top_pair['corr']:.2f})")
                if sc_fig:
                    st.plotly_chart(sc_fig, use_container_width=True)
                else:
                    st.info("Scatter could not be created for the top pair.")
            except Exception as e:
                st.write("Could not create scatter:", e)
        else:
            st.info("No numeric column pairs for correlation analysis.")

        heat = correlation_heatmap_fig(df)
        if heat:
            st.plotly_chart(heat, use_container_width=True)

        if use_method == "IQR (default)":
            out_counts, out_indices = detect_outliers_iqr(df)
        else:
            out_counts, out_indices = detect_outliers_zscore(df, z_thresh=z_threshold)

        out_count_series = pd.Series(out_counts).sort_values(ascending=False)
        st.subheader("🚨 Outlier counts (top 10)")
        if out_count_series.empty:
            st.write("No numeric columns found to detect outliers.")
        else:
            st.table(out_count_series.head(10).rename("count").to_frame())

        top_out_col = out_count_series.index[0] if len(out_count_series)>0 else None
        if top_out_col and out_count_series.iloc[0] > 0:
            try:
                fig = px.box(df, y=top_out_col, points="outliers", title=f"Boxplot: {top_out_col}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write("Could not draw boxplot:", e)

        rule_insights = concise_rule_insights(df, top_corrs, out_counts if 'out_counts' in locals() else {})
        st.subheader("💡 Concise Findings (rule-based)")
        for i, ins in enumerate(rule_insights, start=1):
            st.write(f"{i}. {ins}")

        # NOTE: LLM-polished summary moved to final page (Data Statistics Visualizer).
        raw_report = "AUTO ANALYSIS REPORT\n\n" + "\n".join(rule_insights) + "\n\nTop correlations:\n" + (
            "\n".join([f"{t['x']} - {t['y']}: r={t['corr']:.3f}" for t in top_corrs]) if top_corrs else "None"
        )
        st.download_button("Download Auto Analysis (TXT)", raw_report, "auto_analysis.txt")
        st.download_button("Download Auto Analysis (PDF)", text_to_pdf_bytes(raw_report), "auto_analysis.pdf", mime="application/pdf")

# Final small navigation note (no "Next" action performed — just instruction)
st.markdown("---")
st.info("When you're ready for the consolidated, LLM-polished summary and final export, go to the **Data Statistics Visualizer** page (last page) — it contains the final summary + download buttons.")
