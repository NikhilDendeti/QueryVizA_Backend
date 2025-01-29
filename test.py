import pandas as pd
import plotly.graph_objs as go
import json

# Mock recommendation function (replacing external API)
def mock_chart_recommendation(data: pd.DataFrame) -> dict:
    """Mock chart recommendation for testing purposes."""
    # Example recommendation based on a sample dataset structure
    return {
        "chart_type": "pie",  # Example: Change this to "bar", "line", etc., for testing
        "x_axis": None,
        "y_axis": None,
        "label_column": "gender",  # Replace with column name in your dataset
        "value_column": "id",     # Replace with column name in your dataset
        "rationale": "A pie chart is the best choice to visualize the distribution of genders in the dataset.",
    }

# Generate Plotly chart based on recommendation
def generate_plotly_chart(data: pd.DataFrame, chart_recommendation: dict) -> go.Figure:
    chart_type = chart_recommendation.get("chart_type")
    label_column = chart_recommendation.get("label_column")
    value_column = chart_recommendation.get("value_column")

    if chart_type == "pie":
        if not label_column or not value_column:
            raise ValueError("Both label_column and value_column must be specified for a pie chart.")
        fig = go.Figure(data=[go.Pie(labels=data[label_column], values=data[value_column])])
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    fig.update_layout(
        title=f"{chart_type.capitalize()} Visualization",
        template="plotly_white"
    )
    return fig

def main():
    # Load dataset from a local CSV file
    csv_path = '/home/nikhil/Projects/QueryVizA_Backend/QueryVizAI/MOCK_DATA.csv'
    data = pd.read_csv(csv_path)

    # Ensure dataset is loaded correctly
    if data.empty:
        print("The dataset is empty.")
        return

    # Get chart recommendation
    chart_recommendation = mock_chart_recommendation(data)

    # Generate and display the chart
    try:
        fig = generate_plotly_chart(data, chart_recommendation)
        fig.show()
    except Exception as e:
        print(f"Error generating chart: {e}")

if __name__ == "__main__":
    main()
