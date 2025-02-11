import graphviz

# Create a directed graph for the flowchart
flowchart = graphviz.Digraph(format="png")

# Define the steps in the process
flowchart.node("A", "User Uploads File or Query", shape="parallelogram", style="filled", fillcolor="lightblue")

# File Processing Path
flowchart.node("B", "Process Uploaded File (CSV/PDF)", shape="rectangle", style="filled", fillcolor="lightgray")
flowchart.node("C", "Check File Type (CSV or PDF)", shape="diamond", style="filled", fillcolor="lightcoral")

# PDF Processing
flowchart.node("D", "Extract Text from PDF", shape="rectangle", style="filled", fillcolor="lightgray")

# CSV Processing
flowchart.node("E", "Load CSV & Store in Memory DB", shape="rectangle", style="filled", fillcolor="lightgray")
flowchart.node("F", "Initialize Model (Groq/Gemini)", shape="rectangle", style="filled", fillcolor="lightgray")

# Query Handling
flowchart.node("G", "Analyze Query & Rephrase", shape="rectangle", style="filled", fillcolor="lightgray")
flowchart.node("H", "Generate SQL Query for Data", shape="rectangle", style="filled", fillcolor="lightgray")
flowchart.node("I", "Execute SQL Query", shape="rectangle", style="filled", fillcolor="lightgray")

# Data Analysis & Visualization
flowchart.node("J", "Analyze Data & Recommend Charts", shape="rectangle", style="filled", fillcolor="lightgray")
flowchart.node("K", "Generate Plotly Charts", shape="rectangle", style="filled", fillcolor="lightgray")

# Response Handling
flowchart.node("L", "Return JSON Response (Data/Charts)", shape="parallelogram", style="filled", fillcolor="lightblue")
flowchart.node("M", "Display Results to User", shape="parallelogram", style="filled", fillcolor="lightgreen")

# Define edges for the flowchart
flowchart.edge("A", "B")
flowchart.edge("B", "C")

# Decision point for file type
flowchart.edge("C", "D", label="PDF")
flowchart.edge("C", "E", label="CSV")

# PDF Processing Path
flowchart.edge("D", "L")

# CSV Processing Path
flowchart.edge("E", "F")
flowchart.edge("F", "G")
flowchart.edge("G", "H")
flowchart.edge("H", "I")
flowchart.edge("I", "L")

# Data Analysis & Visualization Path
flowchart.edge("E", "J")
flowchart.edge("J", "K")
flowchart.edge("K", "L")

# Response and Display
flowchart.edge("L", "M")

# Render and display the flowchart
flowchart_path = "flowchart_output"  
flowchart.render(flowchart_path, format="png", cleanup=False)

# Display the flowchart
from IPython.display import display
from PIL import Image

display(Image.open(flowchart_path + ".png"))
