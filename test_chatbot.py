import os
import django
import sqlite3
import pandas as pd
from django.db import connection
from llama_index.llms.groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
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


def rephrase_query(user_query):
    """
    Rephrases a user query into a more structured and formal version.
    """
    try:
        # Use the Groq Llama model to rephrase the query
        response = llm.complete(
            prompt=f"""
            Rephrase the following user query into a more formal and structured version:
            {user_query}.
            """,
            max_tokens=50
        )
        rephrased_query = response.text.strip()
        print("\nRephrased User Query:")
        print(rephrased_query)
        return rephrased_query
    except Exception as e:
        print(f"Error in rephrasing query: {str(e)}")
        # Return the original query if rephrasing fails
        return user_query
    
def chatbot_query(user_query):
    """
    Processes a user query by rephrasing, generating an SQL query, and executing it.
    """
    try:
        # Step 1: Rephrase the user's query
        rephrased_query = rephrase_query(user_query)

        # Step 2: Use the Groq Llama model to generate an SQL query
        response = llm.complete(
            prompt=f"""
            Generate a valid SQL query for the following user query. 
            Assume the users table has columns: {', '.join(data.columns)}. 
            Only return the raw SQL query without any additional text or Markdown formatting:
            {rephrased_query}.
            """,
            max_tokens=100
        )
        # Extract the generated SQL query
        generated_query = response.text.strip()

        # Clean up the response to remove any Markdown formatting
        if generated_query.startswith("```") and generated_query.endswith("```"):
            generated_query = generated_query.strip("```").strip("sql").strip()

        print("\nGenerated SQL Query:")
        print(generated_query)

        # Step 3: Validate and execute the SQL query
        cursor = connection.cursor()
        cursor.execute(generated_query)
        # Fetch column names from the cursor
        columns = [col[0] for col in cursor.description]
        # Fetch all results and format them as dictionaries
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()

        # Step 4: Return the query results
        return {"success": True, "data": results}

    except Exception as e:
        # Print the error for debugging
        print(f"\nError: {str(e)}")
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


# Test the function
if __name__ == "__main__":
    user_query = "Retrieve the top 10 unique email domains with the most male users whose id is an even number, whose IP address contains '20.', and whose last names start with a letter between 'A' and 'M'. Sort by the total count of users for each email domain in descending order, and include the domain, the count of users, and the first id in each group"
    response = chatbot_query(user_query)

    print("\nResponse:")
    print(response)
