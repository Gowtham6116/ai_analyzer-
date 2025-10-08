# 🤖 AI-Based Automated Data Analyzer & Pattern Finder

An intelligent data analysis web app built with **Streamlit**, **Python**, and **Google Gemini AI**.  
It automatically analyzes uploaded datasets, finds patterns, detects anomalies, generates visual charts, and explains insights using AI.

---

## 🚀 Features
- Upload files (CSV, Excel, PDF, DOCX, TXT)
- Automatic data parsing and cleaning
- AI-based insights and summaries using Google Gemini
- Correlation and outlier detection (IQR & Z-Score)
- Smart AI chart generator
- Interactive graphs (Bar, Line, Pie, Histogram, Heatmap)
- Downloadable reports in TXT, CSV, Excel, and PDF

---

## 🧠 Tech Stack
- **Language:** Python  
- **Framework:** Streamlit  
- **AI Engine:** Google Gemini (LLM)  
- **Libraries:** Pandas, NumPy, Plotly, ReportLab, SciPy, python-docx, PyPDF2

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ai_analyzer.git
cd ai_analyzer

# 2. Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate     # on Mac/Linux
venv\Scripts\activate        # on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
Create a file named .env and add:
GENAI_API_KEY=your_api_key_here

# 5. Run the app
streamlit run streamlit_app.py
