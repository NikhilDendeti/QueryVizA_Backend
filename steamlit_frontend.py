# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# import requests
# import json

# # Backend API URL
# BACKEND_URL = "http://10.17.85.46:8000"  # Replace with your backend URL

# def load_and_validate_data(uploaded_file) -> pd.DataFrame:
#     """Load and validate the uploaded CSV file."""
#     try:
#         df = pd.read_csv(uploaded_file)
#         if df.empty:
#             st.error("The uploaded file is empty.")
#             return None
#         return df
#     except Exception as e:
#         st.error(f"Error reading the file: {str(e)}")
#         return None

# def process_file_with_backend(file, model_name: str, source: str) -> dict:
#     """Send file to the backend for processing with improved prompts."""
#     try:
#         # Prepare the file and data for the request
#         files = {"file": ("uploaded_file.csv", file.getvalue(), "text/csv")}

#         # Enhanced prompt to improve results
#         detailed_prompt = (
#             f"Analyze the data in the uploaded CSV file and generate an insightful visualization. "
#             f"Identify the most relevant columns for X-axis and Y-axis based on the data types and content. "
#             f"Provide a meaningful visualization type (e.g., bar chart, line graph, scatter plot) "
#             f"that best represents the relationships or trends in the data. "
#             f"Make sure to include clear labels for axes and a descriptive title for the chart. "
#             f"Provide any additional insights or recommendations for improving the data visualization."
#         )

#         data = {"model_name": model_name, "source": source, "prompt": detailed_prompt}
#         if source in ["Hugging Face", "Meta", "Mistral AI"]:
#             data["source"] = "Groq"
#         else:
#             data["source"] = "Gemini"

#         # Send request to the backend
#         response = requests.post(f"{BACKEND_URL}/process-file/", files=files, data=data)

#         # Handle response
#         if response.status_code == 200:
#             return response.json()
#         else:
#             error_msg = response.json().get("error", "Unknown error")
#             st.error(f"Backend error: {error_msg}")
#             return None
#     except Exception as e:
#         st.error(f"Error communicating with backend: {str(e)}")
#         return None

# def display_plotly_chart(chart_json: str):
#     """Render Plotly chart from JSON."""
#     try:
#         # Load the JSON string into a Plotly figure
#         fig = go.Figure(json.loads(chart_json))
#         st.plotly_chart(fig, use_container_width=True)
#     except Exception as e:
#         st.error(
#             "Error rendering the chart. Ensure the dataset contains numeric or categorical data suitable for visualization."
#         )
#         st.error(f"Technical Details: {str(e)}")

# def main():
#     # Page configuration
#     st.set_page_config(page_title="AI-Powered Visualization", layout="wide")

#     # App title and description
#     st.title("AI-Powered Visualization")
#     st.markdown(
#         """
#         Upload your CSV file and leverage AI to generate interactive data visualizations.
#         Select a language model to process your data and receive insightful charts and recommendations.
#         """
#     )

#     # File upload
#     uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
#     if uploaded_file:
#         # Validate and preview data
#         df = load_and_validate_data(uploaded_file)
#         if df is not None:
#             if st.checkbox("Show data preview"):
#                 st.write(df.head())
#                 st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")

#             # Model selection
#             st.subheader("Model Selection")
#             model_options = {
#                 "Hugging Face": ["distil-whisper-large-v3-en"],
#                 "Meta": [
#                     "llama-3.1-8b-instant",
#                     "llama-3.2-11b-vision-preview",
#                     "llama-3.3-70b-specdec",
#                     "llama-3.3-70b-versatile",
#                 ],
#                 "Mistral AI": ["mixtral-8x7b-32768"],
#                 "Gemini": [
#                     "Gemini 1.5 Pro",
#                     "Gemini 2.0 Flash Experimental",
#                     "LearnLM 1.5 Pro Experimental",
#                 ],
#             }

#             model_group = st.selectbox("Select Model Group", list(model_options.keys()))
#             if model_group:
#                 selected_model = st.selectbox(
#                     "Select a Language Model", model_options[model_group]
#                 )

#                 if selected_model:
#                     st.write(f"Selected Model: **{selected_model}**")

#                     # Generate visualization
#                     if st.button("Generate Visualization"):
#                         with st.spinner("Processing your data..."):
#                             result = process_file_with_backend(uploaded_file, selected_model, model_group)
#                             if result and result.get("success"):
#                                 st.subheader("AI Recommendation")
#                                 st.write(result.get("recommendation", "No recommendation provided."))

#                                 # Display the chart
#                                 if "chart" in result:
#                                     st.subheader("Generated Visualization")
#                                     display_plotly_chart(result["chart"])

#                                 # Display summary statistics
#                                 if st.checkbox("Show Summary Statistics"):
#                                     st.subheader("Summary Statistics")
#                                     st.write(df.describe())
#                             else:
#                                 st.error("Failed to generate visualization. Please try again.")

# if __name__ == "__main__":
#     main()




import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import traceback

BACKEND_URL = "http://10.17.85.48:8000"  # Replace with your backend URL

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

        if source == "Gemini":
            data["model_name"] = model_name.replace(" ", "-").lower()

        response = requests.post(f"{BACKEND_URL}/process-file/", files=files, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            error_msg = response.json().get("error", "Unknown error")
            st.error(f"Backend error: {error_msg}")
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
        else:
            raise ValueError("The provided JSON does not contain valid 'data' and 'layout' keys.")

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        traceback.print_exc()
        st.error("Error rendering the chart. Ensure the dataset contains numeric or categorical data suitable for visualization.")
        st.error(f"Technical Details: {str(e)}")

def main():
    st.set_page_config(page_title="AI-Powered Visualization", layout="wide")

    st.title("AI-Powered Visualization")
    st.markdown(
        """
        Upload your CSV file and leverage AI to generate interactive data visualizations.
        Select a language model to process your data and receive insightful charts and recommendations.
        """
    )

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = load_and_validate_data(uploaded_file)
        if df is not None:
            if st.checkbox("Show data preview"):
                st.write(df.head())
                st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")

            st.subheader("Model Selection")
            model_options = {
                "Meta": [
                    "llama-3.1-8b-instant",
                    "llama-3.2-11b-vision-preview",
                    "llama-3.3-70b-specdec",
                    "llama-3.3-70b-versatile",
                ],
                "Gemini": [
                    "Gemini 1.5 Pro",
                    "Gemini 2.0 Flash Experimental",
                    "LearnLM 1.5 Pro Experimental",
                ]
            }

            model_group = st.selectbox("Select Model Group", list(model_options.keys()))
            if model_group:
                selected_model = st.selectbox(
                    "Select a Language Model", model_options[model_group]
                )

                if selected_model:
                    st.write(f"Selected Model: **{selected_model}**")

                    if st.button("Generate Visualization"):
                        with st.spinner("Processing your data..."):
                            result = process_file_with_backend(uploaded_file, selected_model, model_group)
                            if result and result.get("success"):
                                st.subheader("AI Recommendations")
                                recommendations = result.get("recommendations", [])
                                charts = result.get("charts", [])

                                for idx, (recommendation, chart) in enumerate(zip(recommendations, charts)):
                                    st.markdown(f"### Recommendation {idx + 1}")
                                    st.write(recommendation)

                                    st.markdown(f"#### Chart {idx + 1}")
                                    display_plotly_chart(chart["chart"])

                                if st.checkbox("Show Summary Statistics"):
                                    st.subheader("Summary Statistics")
                                    st.write(df.describe())
                            else:
                                st.error("Failed to generate visualization. Please try again.")

if __name__ == "__main__":
    main()