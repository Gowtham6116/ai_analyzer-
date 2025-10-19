# pages/1_Advanced_Insights.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Advanced Insights", layout="wide")
st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)

st.title("📊 Advanced Data Insights")
st.caption("Automatic data overview, statistical summary, and visual patterns — uses the global dataset (uploaded once).")

# No uploader here — require global df
if 'df' not in st.session_state:
    st.warning("No global dataset found. Upload your data on the 'Upload & Clean Data (Global)' page first.")
    st.stop()

df = st.session_state['df']

st.subheader("🔍 Quick dataset info")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{df.shape[0]}")
c2.metric("Columns", f"{df.shape[1]}")
c3.metric("Missing Values", f"{int(df.isna().sum().sum())}")

st.markdown("---")
# ---- Summary Stats
st.subheader("📈 Summary Statistics")
st.dataframe(df.describe(include="all").transpose())

# ---- Column Distributions (changed graphs)
st.subheader("📊 Value Distributions (Advanced)")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Numeric: show violin (or KDE) for up to first 5 numeric columns
with st.expander("🔢 Numeric Columns Distribution (Violin / KDE)"):
    if not num_cols:
        st.info("No numeric columns detected.")
    else:
        for col in num_cols[:5]:
            try:
                fig = px.violin(df, y=col, box=True, points="outliers", title=f"Violin: {col}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                # fallback histogram
                fig = px.histogram(df, x=col, nbins=30, marginal="box", title=f"Distribution: {col}")
                st.plotly_chart(fig, use_container_width=True)

# Categorical: show treemap for hierarchical view or top bar chart
with st.expander("🔠 Categorical Columns Distribution (Treemap / Top-bar)"):
    if not cat_cols:
        st.info("No categorical columns detected.")
    else:
        for col in cat_cols[:5]:
            try:
                counts = df[col].fillna("(missing)").value_counts().reset_index()
                counts.columns = [col, "count"]
                # If many categories, show top 10 as bar + treemap preview
                top_counts = counts.head(10)
                bar = px.bar(top_counts, x=col, y="count", title=f"Top values in {col}")
                st.plotly_chart(bar, use_container_width=True)
                if len(counts) > 1:
                    try:
                        tree = px.treemap(top_counts, path=[col], values="count", title=f"Treemap of top {col} values")
                        st.plotly_chart(tree, use_container_width=True)
                    except Exception:
                        pass
            except Exception as e:
                st.write(f"Could not draw distribution for {col}: {e}")

st.markdown("---")
# Correlation heatmap (if numeric)
if len(num_cols) >= 2:
    st.subheader("🧩 Correlation Heatmap")
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough numeric columns for correlation heatmap.")

# Missing Values map (small)
st.subheader("🩶 Missing Value Map (compact)")
try:
    # using Plotly to avoid seaborn/matplotlib heavy usage
    missing = df.isnull().astype(int)
    # create heatmap-like matrix sample for display (limit rows)
    sample_miss = missing.head(200)
    fig = px.imshow(sample_miss.T, labels=dict(x="rows sample", y="columns"), title="Missing values (1=missing)")
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.write("Missing value map could not be displayed.")

# Automatic short summary (kept short here; final polished summary moved to final page)
st.markdown("---")
st.subheader("🧠 Quick Auto Insights (rule-based)")
summary = []
summary.append(f"The dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
summary.append(f"There are **{len(num_cols)} numeric** and **{len(cat_cols)} categorical** columns.")
if df.isna().sum().sum() > 0:
    summary.append(f"⚠️ Missing values detected in **{(df.isna().any()).sum()}** columns.")
else:
    summary.append("✅ No missing values detected.")
if len(num_cols) >= 2:
    top_corr = df[num_cols].corr().abs().unstack().sort_values(ascending=False)
    top_corr = top_corr[top_corr < 1].head(1)
    if not top_corr.empty:
        a, b = top_corr.index[0]
        r = top_corr.iloc[0]
        summary.append(f"Strongest correlation is between **{a}** and **{b}** (r = {r:.2f}).")
st.markdown("\n".join([f"- {i}" for i in summary]))

st.info("For the full, LLM-polished summary and downloadable final report, go to the **Data Statistics Visualizer** page (final page).")
