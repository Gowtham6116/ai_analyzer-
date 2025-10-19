# pages/0_Global_Upload.py

import streamlit as st
import pandas as pd
import io, csv, re, numpy as np

st.set_page_config(page_title="Upload & Clean Data", layout="wide")
st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)
st.title("📥 Upload & Clean Data (Global)")
st.caption("Upload once here — the cleaned dataset will be available to all pages automatically.")

def detect_delimiter(sample_text):
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text[:4096])
        return dialect.delimiter
    except Exception:
        return ','

def parse_money_to_cr(val):
    """Parse money-like strings into numeric crores (float). Handles 'Cr', 'Lakh', numbers, commas."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == "" or s.lower() in ["nan", "none", "retained", "retained "]:
        return np.nan
    # Remove currency rupee sign and whitespace
    s = s.replace("₹", "").replace("Rs.", "").replace("rs.", "").strip()
    s = s.replace(",", "")
    # try detect cr/lakh patterns
    m = re.search(r'([\d\.]+)\s*(cr|crore|lakhs?|lakh|lacs?)', s, flags=re.I)
    if m:
        num = float(m.group(1))
        unit = m.group(2).lower()
        if 'cr' in unit or 'crore' in unit:
            return float(num)
        else:
            return float(num) / 100.0
    # plain number
    if re.match(r'^[\d\.]+$', s):
        try:
            return float(s)
        except:
            return np.nan
    m2 = re.search(r'[\d\.]+', s)
    if m2:
        try:
            return float(m2.group(0))
        except:
            return np.nan
    return np.nan

def load_and_clean_dataframe(bytes_data, filename):
    # Try Excel first
    try:
        if filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(bytes_data))
            return df
    except Exception:
        pass
    # try CSV with delimiter detection
    text = bytes_data.decode('utf-8', errors='replace')
    sep = detect_delimiter(text)
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine='python')
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(bytes_data))
        except Exception:
            df = pd.read_csv(io.StringIO(text), sep=',', engine='python', error_bad_lines=False)
    # CLEANING: normalize money-like columns
    money_keywords = ['price', 'cost', '₹', 'in ₹', 'cr', 'lakh', 'amount']
    candidate_cols = [c for c in df.columns if any(k in c.lower() for k in money_keywords)]
    for col in candidate_cols:
        try:
            df[col] = df[col].apply(parse_money_to_cr)
        except Exception:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for c in df.columns:
        if df[c].dtype == object:
            series_numeric = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.extract(r'([\d\.]+)')[0], errors='coerce')
            if series_numeric.notna().sum() / max(1, len(df)) > 0.6:
                df[c] = series_numeric
    return df

# UI - upload
uploaded = st.file_uploader("Upload CSV / Excel (use this Upload page once)", type=["csv", "xlsx", "xls"], key="global_uploader")

if uploaded:
    b = uploaded.read()
    try:
        df = load_and_clean_dataframe(b, uploaded.name)
        st.session_state['df'] = df
        st.session_state['uploaded_file_name'] = uploaded.name
        st.success(f"✅ Loaded and cleaned: {uploaded.name} (Rows: {df.shape[0]}, Cols: {df.shape[1]})")
        st.dataframe(df.head(10))
        st.markdown("**Detected numeric columns:** " + ", ".join(df.select_dtypes(include=['number']).columns.tolist()))
        st.markdown("**If some money columns look incorrect, edit column names or re-upload with consistent formatting.**")
    except Exception as e:
        st.error(f"Failed to load file: {e}")

# show existing dataframe if already uploaded
if 'df' in st.session_state:
    st.markdown("---")
    st.subheader("Currently loaded dataset (global)")
    st.write(f"**File:** {st.session_state.get('uploaded_file_name','(unknown)')}")
    st.dataframe(st.session_state['df'].head(10))
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Clear uploaded dataset"):
            for k in ['df','uploaded_file_name']:
                if k in st.session_state: del st.session_state[k]
            st.experimental_rerun()
    with col2:
        st.markdown("➡️ After upload, go to **AI Analyzer** (main page) to ask questions, generate charts, and run auto analysis.")
else:
    st.info("Upload a CSV or Excel file here. This dataset will be available to all pages (no need to upload again).")
