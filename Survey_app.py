import streamlit as st
from groq import Groq
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats
import json
from fpdf import FPDF
import os
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import uuid
import time
import base64
from werkzeug.utils import secure_filename

print("Starting survey_app.py...")

# Load environment variables
print("Loading .env file...")
load_dotenv()
api_key = os.getenv('GROQ_API_KEY', 'your_api_key_here')
print(f"Loaded API Key: {api_key[:4]}...{api_key[-4:]}")  # Masked for security

# Groq API configuration
try:
    client = Groq(api_key=api_key)
    print("Groq client initialized successfully")
except Exception as e:
    print(f"Failed to initialize Groq client: {str(e)}")
    st.error(f"Failed to initialize Groq client: {str(e)}")

# Configuration
TEMP_DIR = "temp"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
    print(f"Created temp directory: {TEMP_DIR}")

# Session state for audit logs and navigation
if 'audit_logs' not in st.session_state:
    st.session_state.audit_logs = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'file' not in st.session_state:
    st.session_state.file = None
if 'config' not in st.session_state:
    st.session_state.config = {
        'imputation_method': 'median',
        'outlier_method': 'zscore',
        'weight_column': '',
        'schema': {},
        'rules': []
    }
if 'result' not in st.session_state:
    st.session_state.result = None
if 'error' not in st.session_state:
    st.session_state.error = ''
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Upload'

def log_action(action):
    """Log user actions with timestamp and session ID."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    session_id = st.session_state.session_id
    st.session_state.audit_logs.append(f"[{timestamp}] [Session: {session_id}] {action}")

def query_groq(prompt):
    """Query Groq API for suggestions or summarization."""
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            stream=False
        )
        log_action("Groq API called successfully")
        return completion.choices[0].message.content.strip()
    except Exception as e:
        log_action(f"Groq API error: {str(e)}")
        print(f"Groq API error: {str(e)}")
        return None

def suggest_cleaning_methods(df):
    """Use Groq to suggest cleaning methods."""
    missing_percentage = df.isnull().mean().mean() * 100
    numeric_cols = df.select_dtypes(include=np.number).columns
    skewness = df[numeric_cols].skew().mean() if not numeric_cols.empty else 0
    prompt = f"You are a data analysis expert. The dataset has {missing_percentage:.2f}% missing values and average skewness of {skewness:.2f}. Suggest an imputation method (mean, median, KNN) and outlier detection method (IQR, Z-score) with a brief rationale."
    suggestion = query_groq(prompt)
    return suggestion if suggestion else "Using default median imputation and Z-score outlier detection."

def generate_recommendations(df, stats, correlation_matrix):
    """Generate concise AI-driven recommendations based on visualizations."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    skewness = df[numeric_cols].skew().to_dict() if not numeric_cols.empty else {}
    prompt = (
        f"You are a data analysis expert. "
        f"Given these weighted statistics: {json.dumps(stats) if stats else 'None'}, "
        f"skewness: {json.dumps(skewness) if skewness else 'None'}, "
        f"and correlation matrix: {json.dumps(correlation_matrix.to_dict()) if not correlation_matrix.empty else 'None'}, "
        f"give 1-2 short, actionable recommendations for survey data analysis. Be precise."
    )
    recommendations = query_groq(prompt)
    return recommendations if recommendations else "No specific recommendations available due to limited numeric data."

def clean_data(df, config):
    """Perform data cleaning with AI suggestions."""
    log_action("Starting data cleaning")
    print("Cleaning data...")
    
    if not config.get('imputation_method') or not config.get('outlier_method'):
        suggestion = suggest_cleaning_methods(df)
        log_action(f"AI Suggestion: {suggestion}")
        st.info(f"AI Suggestion: {suggestion}")
        if not config.get('imputation_method'):
            config['imputation_method'] = 'median'
        if not config.get('outlier_method'):
            config['outlier_method'] = 'zscore'
    
    imputation_method = config.get('imputation_method')
    if imputation_method == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    elif imputation_method == 'median':
        df = df.fillna(df.median(numeric_only=True))
    elif imputation_method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    log_action(f"Applied {imputation_method} imputation")

    outlier_method = config.get('outlier_method')
    numeric_cols = df.select_dtypes(include=np.number).columns
    if outlier_method == 'iqr':
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
    elif outlier_method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            df.loc[df[col].notna(), col] = df[col].where(z_scores < 3, np.nan)
            df[col] = df[col].fillna(df[col].median())
    log_action(f"Applied {outlier_method} outlier detection")

    for rule in config.get('rules', []):
        col = rule['column']
        condition = rule['condition']
        if condition == 'positive':
            df[col] = df[col].clip(lower=0)
            log_action(f"Applied rule: {col} must be positive")
    
    return df

def apply_weights(df, weight_col):
    """Apply survey weights and compute summaries."""
    if weight_col and weight_col not in df.columns:
        log_action(f"Error: Weight column {weight_col} not found")
        return df, None
    
    weighted_stats = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if weight_col and col != weight_col:
            try:
                weighted_mean = np.average(df[col].dropna(), weights=df[weight_col][df[col].notna()])
                margin_error = 1.96 * np.sqrt(np.var(df[col].dropna() * df[weight_col][df[col].notna()]) / len(df[col].dropna()))
                weighted_stats[col] = {'mean': weighted_mean, 'margin_error': margin_error}
            except Exception as e:
                log_action(f"Error computing weights for {col}: {str(e)}")
                print(f"Error computing weights for {col}: {str(e)}")
    
    log_action("Applied survey weights and computed statistics")
    return df, weighted_stats

class PDFWithWatermark(FPDF):
    def header(self):
        # Save current state
        self.set_font('Arial', 'B', 36)
        self.set_text_color(200, 200, 200)
        self.rotate(30, x=self.w/2, y=self.h/2)
        self.text(self.w/2 - 60, self.h/2, "Survey Data Processor")
        self.rotate(0)
        # Restore color for normal content
        self.set_text_color(0, 0, 0)

    # Add rotate method to FPDF if not present
    def rotate(self, angle, x=None, y=None):
        from math import cos, sin, radians
        if angle != 0:
            angle = radians(angle)
            c = cos(angle)
            s = sin(angle)
            if x is None:
                x = self.x
            if y is None:
                y = self.y
            self._out(f'q {c:.5f} {s:.5f} {-s:.5f} {c:.5f} {x * self.k:.2f} {((self.h - y) * self.k):.2f} cm')
        else:
            self._out('Q')

def generate_report(data, stats, correlation_matrix):
    """Generate PDF report with watermark, concise AI summary, visualizations, recommendations, and audit logs."""
    print("Generating report...")
    pdf = PDFWithWatermark()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(0, 10, "Survey Data Processing Report", ln=True, align="C")
    pdf.ln(5)

    # AI Summary
    if stats:
        stats_summary = json.dumps(stats, indent=2)
        prompt = f"Summarize these survey results in 1-2 short sentences: {stats_summary}"
        ai_summary = query_groq(prompt)
        if ai_summary:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "AI-Generated Summary:", ln=True)
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 8, ai_summary)
            log_action("Generated AI summary for report")
            pdf.ln(2)

    # Recommendations
    recommendations = generate_recommendations(data, stats, correlation_matrix)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "AI-Driven Recommendations:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, recommendations)
    pdf.ln(2)

    # Visualizations
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Visualizations:", ln=True)
    pdf.ln(2)
    plot_paths = []

    try:
        numeric_cols = data.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            # Histogram
            plt.figure(figsize=(6, 4))
            sns.histplot(data[numeric_cols[0]], kde=True, color='blue')
            plt.title(f"Distribution of {numeric_cols[0]}")
            hist_path = os.path.join(TEMP_DIR, f"hist_{uuid.uuid4()}.png")
            plt.savefig(hist_path, bbox_inches='tight')
            plt.close()
            pdf.image(hist_path, x=10, w=90)
            pdf.ln(50)  # Move cursor down after image (adjust as needed)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 8, f"Histogram of {numeric_cols[0]}: Mean={data[numeric_cols[0]].mean():.2f}, Median={data[numeric_cols[0]].median():.2f}, Skew={data[numeric_cols[0]].skew():.2f}")
            pdf.ln(5)
            plot_paths.append(hist_path)
            log_action(f"Generated histogram: {os.path.basename(hist_path)}")

            # Box Plot
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=data[numeric_cols], palette='Set2')
            plt.title("Box Plot of Numeric Columns")
            box_path = os.path.join(TEMP_DIR, f"box_{uuid.uuid4()}.png")
            plt.savefig(box_path, bbox_inches='tight')
            plt.close()
            pdf.image(box_path, x=10, w=90)
            pdf.ln(50)  # Move cursor down after image (adjust as needed)
            pdf.set_font("Arial", size=10)
            stats_box = data[numeric_cols].describe().T
            box_info = ", ".join([f"{col} (Median: {stats_box.loc[col, '50%']:.2f}, IQR: {stats_box.loc[col, '75%']-stats_box.loc[col, '25%']:.2f})" for col in numeric_cols])
            pdf.multi_cell(0, 8, f"Box Plot: {box_info}")
            pdf.ln(5)
            plot_paths.append(box_path)
            log_action(f"Generated box plot: {os.path.basename(box_path)}")

        if len(numeric_cols) > 1:
            # Correlation Heatmap
            plt.figure(figsize=(6, 4))
            corr = data[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title("Correlation Heatmap")
            heatmap_path = os.path.join(TEMP_DIR, f"heatmap_{uuid.uuid4()}.png")
            plt.savefig(heatmap_path, bbox_inches='tight')
            plt.close()
            pdf.image(heatmap_path, x=10, w=90)
            pdf.ln(50)  # Move cursor down after image (adjust as needed)
            pdf.set_font("Arial", size=10)
            if not corr.empty:
                max_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
                if not max_corr.empty:
                    top_corr = max_corr.abs().idxmax()
                    pdf.multi_cell(0, 8, f"Highest correlation: {top_corr[0]} & {top_corr[1]} ({corr.loc[top_corr[0], top_corr[1]]:.2f})")
            pdf.ln(5)
            plot_paths.append(heatmap_path)
            log_action(f"Generated correlation heatmap: {os.path.basename(heatmap_path)}")
        else:
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 8, "Not enough numeric columns for correlation heatmap", ln=True)
            log_action("Skipped correlation heatmap: insufficient numeric columns")

    except Exception as e:
        log_action(f"Error generating visualizations: {str(e)}")
        print(f"Error generating visualizations: {str(e)}")
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, f"Error generating visualizations: {str(e)}", ln=True)

    pdf.ln(5)
    # Audit Logs
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Processing Logs:", ln=True)
    pdf.set_font("Arial", size=9)
    for log in st.session_state.audit_logs:
        pdf.multi_cell(0, 6, log)
    pdf.ln(2)

    # Weighted Statistics
    if stats:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Weighted Statistics:", ln=True)
        pdf.set_font("Arial", size=10)
        for col, stat in stats.items():
            pdf.cell(0, 8, f"{col}: Mean = {stat['mean']:.2f}, Margin of Error = {stat['margin_error']:.2f}", ln=True)

    report_path = os.path.join(TEMP_DIR, f"report_{uuid.uuid4()}.pdf")
    try:
        pdf.output(report_path)
        log_action(f"Generated PDF report: {os.path.basename(report_path)}")
    except Exception as e:
        log_action(f"Error generating PDF report: {str(e)}")
        print(f"Error generating PDF report: {str(e)}")
        raise

    return report_path, plot_paths

# Page navigation
def navigate_to(page):
    st.session_state.current_page = page
    st.rerun()

# Pages
if st.session_state.current_page == 'Upload':
    st.title("Survey Data Processor")
    st.markdown("Upload and process survey data with AI-driven cleaning, analysis, and visualizations.")
    st.header("Upload File")
    uploaded_file = st.file_uploader("Upload CSV/Excel File (max 50MB)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.session_state.error = "File size exceeds 50MB limit"
            st.session_state.file = None
        else:
            filename = secure_filename(f"{uuid.uuid4()}_{uploaded_file.name}")
            file_path = os.path.join(TEMP_DIR, filename)
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.session_state.file = {'path': file_path, 'name': uploaded_file.name}
                st.session_state.error = ''
                log_action(f"File uploaded and saved: {filename}")
            except Exception as e:
                st.session_state.error = f"Error saving file: {str(e)}"
                log_action(f"Error saving file: {str(e)}")
                print(f"Error saving file: {str(e)}")
    
    st.markdown("Select a CSV or Excel file containing survey data.")
    if st.session_state.error:
        st.error(st.session_state.error)
    
    if st.button("Next", disabled=not st.session_state.file):
        navigate_to('Configure')

elif st.session_state.current_page == 'Configure':
    st.title("Survey Data Processor")
    st.markdown("Upload and process survey data with AI-driven cleaning, analysis, and visualizations.")
    st.header("Configure Processing")
    
    with st.form("config_form"):
        imputation_method = st.selectbox(
            "Imputation Method",
            ["median", "mean", "knn"],
            index=0,
            help="Median: Replaces missing values with column median. Mean: Uses column average. KNN: Uses nearest neighbors for imputation."
        )
        outlier_method = st.selectbox(
            "Outlier Method",
            ["zscore", "iqr"],
            index=0,
            help="Z-Score: Removes values beyond 3 standard deviations. IQR: Clips outliers based on interquartile range."
        )
        weight_column = st.text_input("Weight Column (optional)", help="Enter the column name for survey weights.")
        
        st.session_state.config['imputation_method'] = imputation_method
        st.session_state.config['outlier_method'] = outlier_method
        st.session_state.config['weight_column'] = weight_column
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Back"):
                if st.session_state.file and os.path.exists(st.session_state.file['path']):
                    try:
                        os.remove(st.session_state.file['path'])
                        log_action(f"Deleted temporary file: {st.session_state.file['name']}")
                    except Exception as e:
                        log_action(f"Error deleting temporary file: {str(e)}")
                        print(f"Error deleting temporary file: {str(e)}")
                navigate_to('Upload')
        with col2:
            if st.form_submit_button("Next"):
                navigate_to('Process')

elif st.session_state.current_page == 'Process':
    st.title("Survey Data Processor")
    st.markdown("Upload and process survey data with AI-driven cleaning, analysis, and visualizations.")
    st.header("Processing Data")
    
    if st.button("Start Processing"):
        if not st.session_state.file:
            st.session_state.error = "Please upload a file first"
        else:
            st.session_state.audit_logs = []
            log_action("New session started")
            progress_bar = st.progress(0)
            
            try:
                # Read file from disk
                file_path = st.session_state.file['path']
                print(f"Reading file: {file_path}")
                if st.session_state.file['name'].endswith('.csv'):
                    df = pd.read_csv(file_path, chunksize=10000).get_chunk()
                else:
                    df = pd.read_excel(file_path)
                progress_bar.progress(30)
                
                # Process data
                if st.session_state.config.get('schema'):
                    df = df.rename(columns=st.session_state.config['schema'])
                    log_action("Applied schema mapping")
                
                df = clean_data(df, st.session_state.config)
                progress_bar.progress(60)
                
                weight_col = st.session_state.config.get('weight_column')
                df, stats = apply_weights(df, weight_col)
                
                # Compute correlation matrix
                numeric_cols = df.select_dtypes(include=np.number).columns
                correlation_matrix = df[numeric_cols].corr() if len(numeric_cols) > 1 else pd.DataFrame()
                
                st.session_state.result = {
                    'data': df.head(10).to_dict(),
                    'stats': stats,
                    'correlation_matrix': correlation_matrix
                }
                progress_bar.progress(90)
                
                # Generate report
                report_path, plot_paths = generate_report(df, stats, correlation_matrix)
                with open(report_path, "rb") as f:
                    st.session_state.result['report'] = base64.b64encode(f.read()).decode()
                log_action(f"Generated report: {os.path.basename(report_path)}")
                
                # Clean up
                for path in [file_path] + plot_paths:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            log_action(f"Deleted file: {os.path.basename(path)}")
                        except Exception as e:
                            log_action(f"Error deleting file {os.path.basename(path)}: {str(e)}")
                            print(f"Error deleting file {os.path.basename(path)}: {str(e)}")
                
                progress_bar.progress(100)
                st.session_state.error = ''
                navigate_to('Results')
            except Exception as e:
                log_action(f"Processing error: {str(e)}")
                print(f"Processing error: {str(e)}")
                st.session_state.error = str(e)
                if st.session_state.file and os.path.exists(st.session_state.file['path']):
                    try:
                        os.remove(st.session_state.file['path'])
                        log_action(f"Deleted temporary file: {st.session_state.file['name']}")
                    except Exception as e:
                        log_action(f"Error deleting temporary file: {str(e)}")
                        print(f"Error deleting temporary file: {str(e)}")
    
    if st.session_state.error:
        st.error(st.session_state.error)
    
    if st.button("Back"):
        navigate_to('Configure')

elif st.session_state.current_page == 'Results':
    st.title("Survey Data Processor")
    st.markdown("Upload and process survey data with AI-driven cleaning, analysis, and visualizations.")
    st.header("Results")
    
    if st.session_state.result:
        st.subheader("Data Preview")
        st.dataframe(pd.DataFrame(st.session_state.result['data']))
        
        st.subheader("Dashboards")
        
        numeric_cols = pd.DataFrame(st.session_state.result['data']).select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            # Histogram with KDE
            st.markdown("**Histogram with KDE**")
            st.markdown(
                """
                **Purpose**: Shows the distribution of a numeric column.
                """
            )
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                data_hist = pd.DataFrame(st.session_state.result['data'])[numeric_cols[0]]
                sns.histplot(data_hist, kde=True, color='blue', ax=ax)
                ax.set_title(f"Distribution of {numeric_cols[0]}")
                st.pyplot(fig)
                plt.close(fig)
                log_action("Displayed histogram in Streamlit")
                # Simple AI-driven explanation and insights
                hist_prompt = (
                    f"Explain in simple words what this histogram shows. "
                    f"Stats: mean={data_hist.mean():.2f}, median={data_hist.median():.2f}, skew={data_hist.skew():.2f}. "
                    f"Give 2-3 short bullet points."
                )
                hist_explanation = query_groq(hist_prompt)
                st.markdown("**Histogram Insights:**")
                st.markdown(hist_explanation if hist_explanation else "No insights available.")
            except Exception as e:
                st.error(f"Error displaying histogram: {str(e)}")
                log_action(f"Error displaying histogram: {str(e)}")

            # Box Plot
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                data_box = pd.DataFrame(st.session_state.result['data'])[numeric_cols]
                sns.boxplot(data=data_box, palette='Set2', ax=ax)
                ax.set_title("Box Plot of Numeric Columns")
                st.pyplot(fig)
                plt.close(fig)
                log_action("Displayed box plot in Streamlit")
                # Simple AI-driven explanation and insights
                stats_box = data_box.describe().T
                box_prompt = (
                    f"Explain in simple words what this box plot shows. "
                    f"Stats: {stats_box.to_dict()}. "
                    f"Give 2-3 short bullet points."
                )
                box_explanation = query_groq(box_prompt)
                st.markdown("**Box Plot Insights:**")
                st.markdown(box_explanation if box_explanation else "No insights available.")
            except Exception as e:
                st.error(f"Error displaying box plot: {str(e)}")
                log_action(f"Error displaying box plot: {str(e)}")

            # Correlation Heatmap
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                corr = pd.DataFrame(st.session_state.result['data'])[numeric_cols].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)
                plt.close(fig)
                log_action("Displayed correlation heatmap in Streamlit")
                # Simple AI-driven explanation and insights
                corr_prompt = (
                    f"Explain in simple words what this correlation heatmap shows. "
                    f"Matrix: {corr.to_dict()}. "
                    f"Give 2-3 short bullet points."
                )
                corr_explanation = query_groq(corr_prompt)
                st.markdown("**Correlation Heatmap Insights:**")
                st.markdown(corr_explanation if corr_explanation else "No insights available.")
            except Exception as e:
                st.error(f"Error displaying correlation heatmap: {str(e)}")
                log_action(f"Error displaying correlation heatmap: {str(e)}")
        
        st.subheader("AI Recommendations")
        recommendations = generate_recommendations(pd.DataFrame(st.session_state.result['data']), st.session_state.result['stats'], pd.DataFrame(st.session_state.result['correlation_matrix']))
        st.markdown(recommendations)
        
        st.subheader("Audit Logs")
        for log in st.session_state.audit_logs:
            st.markdown(f"- {log}")
        
        if st.button("Generate New Report"):
            try:
                report_path, plot_paths = generate_report(pd.DataFrame(st.session_state.result['data']), st.session_state.result['stats'], pd.DataFrame(st.session_state.result['correlation_matrix']))
                with open(report_path, "rb") as f:
                    st.session_state.result['report'] = base64.b64encode(f.read()).decode()
                st.success("New report generated successfully")
                log_action(f"Generated new report: {os.path.basename(report_path)}")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                log_action(f"Error generating report: {str(e)}")
        
        if st.session_state.result.get('report'):
            st.download_button(
                "Download Report",
                data=base64.b64decode(st.session_state.result['report']),
                file_name="survey_data_report.pdf",
                mime="application/pdf"
            )

# Footer
st.markdown(
    """
    <style>
    footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        color: #555;
    }
    </style>
    <footer>
    <p>Survey Data Processor &copy; 2025. All rights reserved.</p>
    <p>Powered by Streamlit, Groq, and OpenAI.</p>
    </footer>
    """,
    unsafe_allow_html=True
)

print("survey_app.py execution completed.")
# End of the Streamlit app
# Note: This code is designed to run in a Streamlit environment.
# Ensure you have the required libraries installed and run this script with Streamlit.