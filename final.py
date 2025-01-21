# import os
# import sqlite3
# import pandas as pd
# import plotly.graph_objs as go
# from dotenv import load_dotenv
# from llama_index.llms.groq import Groq

# # Load environment variables
# load_dotenv()

# # Get the Groq API key from the environment
# groq_api_key = os.getenv("GROQ_API_KEY")

# if not groq_api_key:
#     raise EnvironmentError("GROQ_API_KEY is not set. Ensure it is properly configured.")

# # Initialize the Groq Llama model
# llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

# # Load the CSV file into an in-memory SQLite database
# csv_file_path = '/home/nikhil/Projects/QueryVizA_Backend/QueryVizAI/MOCK_DATA.csv'
# data = pd.read_csv(csv_file_path)

# # Create an in-memory SQLite database and load the data
# connection = sqlite3.connect(":memory:")
# data.to_sql("users", connection, if_exists="replace", index=False)


# def rephrase_query(user_query):
#     """Rephrases a user query into a more structured and formal version."""
#     try:
#         response = llm.complete(
#             prompt=f"Rephrase the following user query into a formal and structured version:\n{user_query}.",
#             max_tokens=50
#         )
#         return response.text.strip()
#     except Exception as e:
#         print(f"Error in rephrasing query: {str(e)}")
#         return user_query


# def generate_query(user_query):
#     """Generates an SQLite-compatible SQL query based on the user's query."""
#     try:
#         rephrased_query = rephrase_query(user_query)
#         response = llm.complete(
#             prompt=f"""
#             Generate a valid SQLite-compatible SQL query for the following user query.
#             The users table has columns: {', '.join(data.columns)}.
#             Ensure the query does not include MySQL-specific functions.
            
#             {rephrased_query}.
#             """,
#             max_tokens=200
#         )
#         sql_query = response.text.strip()
#         if "SELECT" in sql_query:
#             sql_query = sql_query[sql_query.find("SELECT"):]
#         print("\nGenerated SQL Query:")
#         print(sql_query)
#         return sql_query
#     except Exception as e:
#         print(f"Error generating SQL query: {str(e)}")
#         return None


# def execute_query(sql_query):
#     """Executes the SQL query and returns the results as a list of dictionaries."""
#     try:
#         cursor = connection.cursor()
#         cursor.execute(sql_query)
#         columns = [col[0] for col in cursor.description]
#         results = [dict(zip(columns, row)) for row in cursor.fetchall()]
#         cursor.close()
#         return {"success": True, "data": results}
#     except Exception as e:
#         print(f"Error executing query: {str(e)}")
#         return {"success": False, "error": str(e)}


# def visualize_query_results(results, x_axis=None, y_axis=None, chart_type=None, color_by=None):
#     """Dynamically visualizes the query results using Plotly."""
#     try:
#         if not results or "data" not in results or not results["data"]:
#             print("No data to visualize.")
#             return

#         df = pd.DataFrame(results["data"])

#         if not x_axis or not y_axis:
#             numeric_columns = df.select_dtypes(include=["number"]).columns
#             categorical_columns = df.select_dtypes(exclude=["number"]).columns

#             if len(numeric_columns) >= 2:
#                 x_axis, y_axis = numeric_columns[:2]
#             elif len(numeric_columns) == 1:
#                 x_axis = numeric_columns[0]
#                 y_axis = categorical_columns[0] if len(categorical_columns) > 0 else numeric_columns[0]
#             else:
#                 x_axis = categorical_columns[0]
#                 y_axis = categorical_columns[1] if len(categorical_columns) > 1 else None

#         if not y_axis:
#             print("Insufficient data to visualize.")
#             return

#         if not chart_type:
#             chart_type = "bar" if df[x_axis].nunique() < 20 else "line"

#         hover_text = [
#             "<br>".join(f"{key}: {value}" for key, value in row.items()) for row in results["data"]
#         ]

#         fig = go.Figure()

#         if chart_type == "bar":
#             fig.add_trace(
#                 go.Bar(
#                     x=df[x_axis],
#                     y=df[y_axis],
#                     marker=dict(
#                         color="blue",
#                         colorscale="Viridis" if not color_by else None
#                     ),
#                     text=df[y_axis],
#                     texttemplate="%{text}",
#                     textposition="outside",
#                     hovertext=hover_text,
#                     hoverinfo="text",
#                 )
#             )
#         elif chart_type == "line":
#             fig.add_trace(
#                 go.Scatter(
#                     x=df[x_axis],
#                     y=df[y_axis],
#                     mode="lines+markers",
#                     marker=dict(color="blue"),
#                     hovertext=hover_text,
#                     hoverinfo="text",
#                 )
#             )

#         fig.update_layout(
#             title=f"{chart_type.capitalize()} Chart of {x_axis.capitalize()} vs {y_axis.capitalize()}",
#             xaxis_title=x_axis.capitalize(),
#             yaxis_title=y_axis.capitalize(),
#             template="plotly_white",
#             height=700,
#             width=1200,
#             margin=dict(l=50, r=50, t=70, b=50),
#             font=dict(family="Arial, sans-serif", size=14, color="black"),
#             xaxis=dict(tickangle=45, tickfont=dict(size=12)),
#             yaxis=dict(tickfont=dict(size=12)),
#         )

#         fig.show()

#     except Exception as e:
#         print(f"Visualization error: {str(e)}")


# user_queries = [
#     "Retrieve all female users whose IP address starts with '192.'",
#     "Get all users whose first name starts with the letter 'J'",
#     "Retrieve all users whose last name ends with 'son' and who have an odd user ID",
#     "Get all users with a Gmail email domain (@gmail.com) and an IP address containing '10.'",
#     "Find all male users regardless of the case in the gender field (e.g., Male, male, MALE)",
#     "Retrieve all users with email domains containing EDU, case-insensitively",
#     "Get the total count of users grouped by gender",
#     "List the top 5 most common email domains and the count of users for each domain",
#     "Retrieve all users sorted by their last name in ascending order",
#     "Get all users sorted by their user ID in descending order",
#     "Find the email domain with the highest number of users and include the count",
#     "Retrieve the top 3 email domains with the most female users",
#     "Retrieve the top 10 unique email domains for users whose IDs are divisible by 3 and whose IP addresses start with '172.'",
#     "Get the total count of male users whose last names start with a letter between 'A' and 'G'",
#     "Retrieve all users with no IP address (test for NULL or empty values in the IP address column)",
#     "Find users with an ID less than 10,000 (testing for nonexistent data in your mock dataset)",
#     "Retrieve all users, displaying their first name, last name, email, and IP address only",
#     "Get all users whose IP address contains '20.' and display their first name, last name, email, and gender",
#     "Retrieve the total count of users grouped by gender and sorted in descending order of count",
#     "List the top 10 unique email domains with the most users who are female, whose last name starts with 'A', and whose user ID is a multiple of 5. Include the domain, the count of users, and the first ID in each group"
# ]

# if __name__ == "__main__":
#     for i, user_query in enumerate(user_queries, start=1):
#         print(f"\n--- Running Query {i} ---")
#         print(f"User Query: {user_query}")

#         sql_query = generate_query(user_query)

#         if sql_query:
#             query_results = execute_query(sql_query)
#             if query_results["success"]:
#                 print(f"Query Results for Query {i}:")
#                 print(query_results["data"])
#                 visualize_query_results(query_results)
#             else:
#                 print(f"Query Execution Failed for Query {i}: {query_results['error']}")
#         else:
#             print(f"Failed to generate SQL query for Query {i}")




import os
import sqlite3
import pandas as pd
import plotly.graph_objs as go
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

# Load environment variables
load_dotenv()

# Get the Groq API key from the environment
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise EnvironmentError("GROQ_API_KEY is not set. Ensure it is properly configured.")

# Initialize the Groq Llama model
llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

# Load the CSV file into an in-memory SQLite database
csv_file_path = '/home/nikhil/Projects/QueryVizA_Backend/QueryVizAI/MOCK_DATA.csv'
data = pd.read_csv(csv_file_path)

# Create an in-memory SQLite database and load the data
connection = sqlite3.connect(":memory:")
data.to_sql("users", connection, if_exists="replace", index=False)

# Function to rephrase the user query
def rephrase_query(user_query):
    try:
        response = llm.complete(
            prompt=f"Rephrase the following user query into a formal and structured SQL-compatible version:\n{user_query}",
            max_tokens=50
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error rephrasing query: {e}")
        return user_query

# Function to generate SQL query
def generate_query(user_query):
    try:
        rephrased_query = rephrase_query(user_query)
        response = llm.complete(
            prompt=f"""
Generate a valid SQL query for SQLite. Assume the users table has the following columns:
{', '.join(data.columns)}.
{rephrased_query}.
""",
            max_tokens=150
        )
        query = response.text.strip()
        if "SELECT" in query:
            query = query[query.find("SELECT"):]
        # Strip semicolons or unexpected trailing SQL
        query = query.split(";")[0]
        return query
    except Exception as e:
        print(f"Error generating query: {e}")
        return None

# Function to execute the SQL query
def execute_query(sql_query):
    try:
        cursor = connection.cursor()
        cursor.execute(sql_query)
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        return {"success": True, "data": results}
    except sqlite3.Error as e:
        return {"success": False, "error": str(e)}

# Function to visualize query results
def visualize_results(results, x_axis=None, y_axis=None):
    if not results["success"] or not results["data"]:
        print("No data to visualize.")
        return

    df = pd.DataFrame(results["data"])
    if not x_axis or not y_axis:
        print("Insufficient data for visualization.")
        return

    fig = go.Figure(data=[go.Bar(x=df[x_axis], y=df[y_axis])])
    fig.update_layout(
        title=f"{x_axis.capitalize()} vs {y_axis.capitalize()}",
        xaxis_title=x_axis.capitalize(),
        yaxis_title=y_axis.capitalize(),
    )
    fig.show()

# List of all 20 queries
user_queries = ["Get all the men"]

# Main Execution
if __name__ == "__main__":
    for i, query in enumerate(user_queries, start=1):
        print(f"\n--- Running Query {i} ---")
        sql_query = generate_query(query)
        if sql_query:
            print(f"Generated SQL Query:\n{sql_query}")
            results = execute_query(sql_query)
            if results["success"]:
                print(f"Results for Query {i}:\n{results['data']}")
                if len(results["data"]) > 0:
                    # Visualize the first two columns of the result
                    keys = list(results["data"][0].keys())
                    if len(keys) >= 2:
                        visualize_results(results, x_axis=keys[0], y_axis=keys[1])
            else:
                print(f"Query Execution Failed for Query {i}: {results['error']}")
        else:
            print(f"Failed to generate SQL query for Query {i}")
