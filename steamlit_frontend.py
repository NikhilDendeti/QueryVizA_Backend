import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import traceback

# Must be the first Streamlit command
st.set_page_config(page_title="AI-Powered Visualization", layout="wide")

# Backend API URL
BACKEND_URL = "http://10.17.85.48:8000"  # Update with the actual backend URL

# Session state initialization
if "df" not in st.session_state:
    st.session_state.df = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None


def load_and_validate_data(uploaded_file) -> pd.DataFrame:
    """Load and validate the uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("The uploaded file is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading the file: {str(e)}")
        return None


def process_file_with_backend(file, model_name: str, source: str) -> dict:
    """Send file to the backend for processing."""
    try:
        files = {"file": ("uploaded_file.csv", file.getvalue(), "text/csv")}
        data = {"model_name": model_name, "source": source}

        response = requests.post(f"{BACKEND_URL}/process-file/", files=files, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend error: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error communicating with backend: {str(e)}")
        return None


def display_plotly_chart(chart_json):
    """Render Plotly chart from JSON or dictionary."""
    try:
        if isinstance(chart_json, str):  
            chart_dict = json.loads(chart_json)  
        elif isinstance(chart_json, dict):  
            chart_dict = chart_json
        else:
            raise ValueError("Invalid input type for chart_json. Must be a JSON string or dictionary.")

        if "data" in chart_dict and "layout" in chart_dict:
            fig = go.Figure(data=chart_dict["data"], layout=chart_dict["layout"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            raise ValueError("The provided JSON does not contain valid 'data' and 'layout' keys.")

    except Exception as e:
        traceback.print_exc()
        st.error("Error rendering the chart. Ensure the dataset contains numeric or categorical data suitable for visualization.")
        st.error(f"Technical Details: {str(e)}")


# Sidebar for file upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    df = load_and_validate_data(uploaded_file)
    if df is not None:
        st.session_state.df = df 

df = st.session_state.df

if df is not None:
    st.sidebar.write(f"✔ Data Loaded ({df.shape[0]} rows, {df.shape[1]} columns)")

st.sidebar.divider()


# **Natural Language Query Section**
st.subheader("🔍 Natural Language Query")

# Define model selection dropdowns
model_options = {
    "Meta": [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768"
    ],
    "Gemini": [
        "gemini-1.5-pro-latest",
        "gemini-1.0-pro-latest"
    ]
}

# Prevents UI errors when switching model groups
model_group = st.selectbox("Select Model Group", list(model_options.keys()), key="model_group")
selected_model = st.selectbox("Select a Language Model", model_options[model_group], key="selected_model") if model_group else None

# Query input
nl_query = st.text_input("Ask a question about your data (e.g., 'Show top 5 sales by region'):")

if nl_query and st.button("Process Query"):
    if st.session_state.uploaded_file and selected_model and model_group:
        with st.spinner("Analyzing your question..."):
            try:
                # Reset file pointer before sending
                st.session_state.uploaded_file.seek(0)

                files = {"file": ("data.csv", st.session_state.uploaded_file, "text/csv")}
                data = {
                    "model_name": selected_model,
                    "source": model_group,
                    "query": nl_query
                }
                
                response = requests.post(f"{BACKEND_URL}/process-query/", files=files, data=data)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        st.subheader("✅ Query Results")

                        if result.get("sql"):
                            st.code(f"Generated SQL: {result['sql']}", language='sql')

                        if result.get("data"):
                            result_df = pd.DataFrame(result['data'])
                            st.dataframe(result_df)

                        if result.get("explanation"):
                            st.markdown(f"**📌 Explanation:** {result['explanation']}")

                        if result.get("charts"):
                            st.subheader("📊 Visualizations")
                            for chart in result["charts"]:
                                display_plotly_chart(chart["chart"])
                    else:
                        st.error(f"Backend error: {result.get('error', 'Unknown error')}")
                else:
                    st.error(f"Request failed ({response.status_code}): {response.text}")

            except Exception as e:
                st.error(f"Client error: {str(e)}")
    else:
        st.error("Please upload a file and select a model before processing the query.")

st.divider()


# **Visualization Generation Section**
st.subheader("📊 Generate AI-Powered Visualization")

if df is not None:
    if st.checkbox("Show Data Preview"):
        st.dataframe(df)
    
    if st.button("Generate AI Visualization"):
        with st.spinner("Processing your data..."):
            result = process_file_with_backend(st.session_state.uploaded_file, selected_model, model_group)

            if result and result.get("success"):
                st.subheader("🎯 AI Recommendations")
                recommendations = result.get("recommendations", [])
                charts = result.get("charts", [])

                for idx, (recommendation, chart) in enumerate(zip(recommendations, charts)):
                    st.markdown(f"### Recommendation {idx + 1}")
                    st.write(recommendation)
                    st.markdown(f"#### Chart {idx + 1}")
                    display_plotly_chart(chart["chart"])

                if st.checkbox("Show Summary Statistics"):
                    st.subheader("📊 Summary Statistics")
                    st.write(df.describe())

            else:
                st.error("❌ Failed to generate visualization. Please try again.")

