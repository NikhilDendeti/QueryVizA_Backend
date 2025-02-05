import os
import pandas as pd
import plotly.graph_objs as go
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import json
from pandasql import sqldf
import re
import sqlite3
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

DEFAULT_CHART_EXAMPLES = """
Here are examples of different Plotly charts:

1. Bar Chart:
```python
fig = go.Figure(data=[go.Bar(x=df['x_column'], y=df['y_column'])])
fig.update_layout(title='Bar Chart Example', xaxis_title='X-Axis', yaxis_title='Y-Axis')
```

2. Line Chart:
```python
fig = go.Figure(data=[go.Scatter(x=df['x_column'], y=df['y_column'], mode='lines')])
fig.update_layout(title='Line Chart Example', xaxis_title='X-Axis', yaxis_title='Y-Axis')
```

3. Scatter Plot:
```python
fig = go.Figure(data=[go.Scatter(x=df['x_column'], y=df['y_column'], mode='markers')])
fig.update_layout(title='Scatter Plot Example', xaxis_title='X-Axis', yaxis_title='Y-Axis')
```

4. Pie Chart:
```python
fig = go.Figure(data=[go.Pie(labels=df['label_column'], values=df['value_column'])])
fig.update_layout(title='Pie Chart Example')
```

5. Heatmap:
```python
fig = go.Figure(data=[go.Heatmap(z=df['z_column'], x=df['x_column'], y=df['y_column'])])
fig.update_layout(title='Heatmap Example', xaxis_title='X-Axis', yaxis_title='Y-Axis')
```
"""

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def initialize_google_model(model_name: str):
    """Initialize Google Generative AI model with proper configuration."""
    try:
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )
        return model.start_chat(history=[])
    except Exception as e:
        raise RuntimeError(f"Error initializing Google model: {str(e)}")
   

def initialize_model(model_name: str, source: str):
    if source in ["Hugging Face","Meta","Mistral AI"]:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    elif source == "Gemini":
        return initialize_google_model(model_name)
    else:
        raise ValueError("Unsupported model selected")

def analyze_and_recommend(data: pd.DataFrame, model_instance, model_name) -> dict:
    try:
        columns = ", ".join(data.columns)
        sample_data = data.head(3).to_dict(orient='records')
        prompt = f"""
        You are a seasoned data analyst. Your task is to provide a comprehensive analysis of the dataset to uncover key insights, trends, and patterns. Focus on delivering actionable business insights and highlighting important data characteristics.
        
        **Dataset Overview:**
        - Columns: {columns}
        - Sample Data: {sample_data}


        **Analysis Guidelines:**
        1. **Data Overview**: Briefly describe the dataset structure and variables
        2. **Key Trends**: Identify significant trends in the data (e.g., increases/decreases over time, category distributions)
        3. **Patterns & Correlations**: Highlight notable correlations between variables
        4. **Outliers/Anomalies**: Point out any unexpected values or inconsistencies
        5. **Business Implications**: Suggest how these insights could impact business decisions
        6. **Data Quality**: Note any missing values or data quality issues
        7. **Visualization Strategy**: Explain how the recommended charts help answer key business questions

        ## **Guidelines for Visualization Selection**
        - **Choose the most appropriate chart types** based on the data distribution and relationships (e.g., bar chart for categorical comparisons, scatter plot for numerical correlations, pie chart for proportions, etc.).
        - **Only include valid column mappings**:
        - Ensure that `x_axis` and `y_axis` are **never null** for charts that require both axes (e.g., bar, line, scatter, heatmap).
        - Ensure that `label_column` and `value_column` are **never null** for pie charts.
        - **Ensure meaningful axis selections**:
        - Do **not** use non-numeric columns for `y_axis` unless appropriate (e.g., counts, aggregations).
        - Avoid using high-cardinality categorical columns as `x_axis` unless meaningful.
        - **If a chart type is not appropriate, DO NOT include it.**
        - Do not include charts that lack the required numerical or categorical relationships.
        - Do not suggest heatmaps or scatter plots if they do not have valid numeric or categorical pairings.
        - If a chart field is missing or not meaningful, **restructure the response** instead of leaving fields `null`.

        Guidelines:
        - Examine the column names and infer their likely data types and relationships.
        - Suggest the best type of chart for visualizing the data (e.g., bar chart, line chart, scatter plot, pie chart, heatmap).
        - For charts requiring axes (e.g., bar, scatter, line), specify the most appropriate X-axis and Y-axis columns .
        - Ensure for which you are recommending the chart type and like wise give the appropriate X-axis and Y-axis ensure you didn't miss anything.
        - Suggest all relevant chart types based on the dataset.
        - Ensure that for each chart type, all required axes and values are specified.
        - If multiple charts can be useful, suggest all relevant ones.
        - Ensure completeness and correctness of the recommendations before responding.
        - Provide a clear explanation (rationale) for why this chart type is the best choice.
        - Suggest any additional columns that can enhance the visualization (e.g., color, size, or labels for better insights).
        - Don't give me any resonings only the give JSON Format.
         Respond strictly in **valid JSON format** as a **single list** of objects.
        - **DO NOT return multiple separate JSON objects.** Instead, return a single JSON array like this:
        - Don't specify json in the response.
        - Retry up to 3 times in case of missing fields before reporting an error.

        Examples of visualizations include:
        {DEFAULT_CHART_EXAMPLES}

        Ensure your response adheres to the following JSON structure strictly:
        ```
        {{
            "chart_type": "string",        // The recommended chart type (e.g., "bar", "line", "scatter", etc.)
            "x_axis": "string or null",    // Column name for the X-axis (or null if not applicable)
            "y_axis": "string or null",    // Column name for the Y-axis (or null if not applicable)
            "label_column": "string or null", // Column name for labels (or null if not applicable)
            "value_column": "string or null", // Column name for values (or null if not applicable)
            "rationale": "string"          // Explanation of why this chart type is recommended
            "Data Analyst": "string"      // Explanation as a Data analyst
        }}
        """

        if isinstance(model_instance, genai.ChatSession):  # Google model
            response = model_instance.send_message(prompt)
            response_content = response.text
        else:
            completion = model_instance.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": prompt}],
                temperature=1,
                max_completion_tokens=1024
            )

            response_content = completion.choices[0].message.content
        print(response_content)
        recommendation = json.loads(response_content)

        if not isinstance(recommendation, list):
            raise ValueError("Response should be a list of chart recommendations")

        required_keys = {"chart_type", "rationale"}

        optional_keys = {"x_axis", "y_axis", "label_column", "value_column"}
        
        validated_recommendations = []
        for idx, recommendation in enumerate(recommendation):
            # Ensure all required keys are present
            if not required_keys.issubset(recommendation.keys()):
                missing = required_keys - recommendation.keys()
                raise ValueError(f"Entry {idx} missing required keys: {', '.join(missing)}")

            # Ensure optional keys are either present or null
            for key in optional_keys:
                if key not in recommendation:
                    recommendation[key] = None  # Set missing optional keys to null

            validated_recommendations.append(recommendation)

        return validated_recommendations

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response: {e}")
    except Exception as e:
        raise RuntimeError(f"Error analyzing and recommending chart: {e}")

def generate_plotly_chart(data: pd.DataFrame, chart_recommendation: dict) -> dict:
    try:
        chart_type = chart_recommendation.get("chart_type")
        x_axis = chart_recommendation.get("x_axis")
        y_axis = chart_recommendation.get("y_axis")
        label_column = chart_recommendation.get("label_column")
        value_column = chart_recommendation.get("value_column")
        rationale = chart_recommendation.get("rationale")

        if x_axis == "null":
            x_axis = None
        if y_axis == "null":
            y_axis = None
        if label_column == "null":
            label_column = None
        if value_column == "null":
            value_column = None

        if not chart_type:
            raise ValueError("No chart type recommended.")

        if chart_type in ["bar", "line", "scatter", "heatmap"] and (not x_axis or not y_axis):
            raise ValueError(f"Both x_axis and y_axis must be specified for a {chart_type} chart.")

        if chart_type == "pie" and (not label_column or not value_column):
            raise ValueError("Both label_column and value_column must be specified for a pie chart.")

        if chart_type == "bar":
            fig = go.Figure(data=[go.Bar(x=data[x_axis], y=data[y_axis])])

        elif chart_type == "line":
            fig = go.Figure(data=[go.Scatter(x=data[x_axis], y=data[y_axis], mode="lines")])

        elif chart_type == "scatter":
            fig = go.Figure(data=[go.Scatter(x=data[x_axis], y=data[y_axis], mode="markers")])

        elif chart_type == "pie":
            fig = go.Figure(data=[go.Pie(labels=data[label_column], values=data[value_column])])

        elif chart_type == "heatmap":
            if not x_axis or not y_axis:
                raise ValueError("Both x_axis and y_axis must be specified for a heatmap.")
            z_values = data.groupby([x_axis, y_axis]).size().unstack(fill_value=0)
            fig = go.Figure(data=[go.Heatmap(z=z_values.values, x=z_values.columns, y=z_values.index)])

        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        fig.update_layout(
            title=f"{chart_type.capitalize()} Visualization",
            xaxis_title=x_axis.capitalize() if x_axis else None,
            yaxis_title=y_axis.capitalize() if y_axis else None,
            template="plotly_white"
        )

        return {
            "recommendation": chart_recommendation,
            "chart": fig.to_json(),
        }

    except KeyError as e:
        raise RuntimeError(f"Missing key in data or recommendation: {e}")
    except Exception as e:
        raise RuntimeError(f"Error generating Plotly chart: {e}")


@app.post("/process-file/")
async def process_file(file: UploadFile = File(...), model_name: str = Form(...), source: str = Form(...)):
    try:
        print(f"Model Name: {model_name}, Source: {source}")
        print(f"File Content Type: {file.content_type}")

        if file.content_type == "application/pdf":
            pdf_reader = PdfReader(file.file)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            return {"success": True, "type": "pdf", "content": text}

        elif file.content_type == "text/csv":
            data = pd.read_csv(file.file)
            if data.empty:
                raise ValueError("Uploaded CSV file is empty.")

            model = initialize_model(model_name, source)

            recommendations = analyze_and_recommend(data, model, model_name)

            charts = []
            for recommendation in recommendations:
                try:
                    chart_json = generate_plotly_chart(data, recommendation)
                    charts.append(chart_json)
                except Exception as e:
                    print(f"Error generating chart for recommendation: {recommendation}. Error: {e}")
                    continue  
            return {
                "success": True,
                "recommendations": recommendations,
                "charts": charts
            }

        else:
            return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    


def extract_valid_json(response_text: str) -> dict:
    """Enhanced JSON extraction with better error handling."""
    print(response_text)
    response_text = response_text.strip()
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass  

    json_match = re.search(
        r'```(?:json)?\s*({.*?})\s*```', 
        response_text, 
        re.DOTALL | re.IGNORECASE
    )
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except Exception as e:
            raise RuntimeError(f"JSON in code block invalid: {str(e)}")

    # Attempt to find JSON-like structure in text
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        return json.loads(response_text[start:end])
    except Exception as e:
        raise RuntimeError(f"Failed to extract JSON: {str(e)}")


def process_uploaded_file(file: UploadFile):
    """
    Processes the uploaded CSV file and loads it into an in-memory SQLite database.
    """
    try:
        data = pd.read_csv(file.file)
        
        connection = sqlite3.connect(":memory:")
        data.to_sql("users", connection, if_exists="replace", index=False)
        
        return data, connection
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

def rephrase_query(user_query: str, llm, model_name):
    """
    Rephrases a user query into a more structured and formal version.
    """
    try:

        prompt=f"""
            Rephrase the following user query into a more formal and structured version:
            {user_query}.
        """

        completion = llm.chat.completions.create(
            model=model_name,
                messages=[{"role": "system", "content": prompt}],
                temperature=1,
                max_completion_tokens=1024
            )

        response_content = completion.choices[0].message.content
        print(response_content)

        
        return response_content.text.strip()
    except Exception as e:
        return user_query

def chatbot_query(user_query: str, connection, data, llm, model_name):
    """
    Processes a user query by rephrasing, generating an SQL query, and executing it.
    """
    try:
        rephrased_query = rephrase_query(user_query, llm, model_name)
        prompt=f"""
            Generate a valid SQL query for the following user query. 
            Assume the users table has columns: {', '.join(data.columns)}. 
            According to the colums take the column names and generate the SQL query properly.
            Only return the raw SQL query without any additional text or Markdown formatting:
            If there are spaces in the column names make sure in the SQL query you are using the square brackets.
            {rephrased_query}.
            """
        completion = llm.chat.completions.create(
            model=model_name,
                messages=[{"role": "system", "content": prompt}],
                temperature=1,
                max_completion_tokens=1024
            )

        response_content = completion.choices[0].message.content
        print(response_content)
        generated_query = response_content.strip()
        print(generated_query)
        if generated_query.startswith("```") and generated_query.endswith("```"):
            generated_query = generated_query.strip("```").strip("sql").strip()

        cursor = connection.cursor()
        cursor.execute(generated_query)
        columns = [col[0] for col in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()

        return {"success": True, "data": results}

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

@app.post("/process-query/")
async def upload_file(
        file: UploadFile = File(...),
    model_name: str = Form(...),
    source: str = Form(...),
    query: str = Form(...)
):
    """
    Endpoint to upload a CSV file and execute a user query.
    """
    llm = initialize_model(model_name=model_name, source=source)
    data, connection = process_uploaded_file(file)
    response = chatbot_query(query, connection, data, llm, model_name)
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8000)




