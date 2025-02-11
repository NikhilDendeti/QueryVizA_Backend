# AI-Powered Data Query and Visualization Platform

## 📌 Project Overview
This project enables AI-driven **data querying** and **visualization** using **FastAPI** as the backend and **Streamlit** as the frontend. It allows users to:
- **Upload CSV files** and process them.
- **Generate SQL queries** from natural language queries.
- **Receive AI-generated data insights and chart recommendations.**
- **Visualize results** using Plotly.

---

## 🚀 Tech Stack
### **Backend (FastAPI)**
- **FastAPI**: Handles API requests.
- **Groq / Gemini AI**: Generates SQL queries and chart recommendations.
- **Pandas / SQLite**: Processes and stores CSV data.
- **PyPDF2**: Extracts text from PDFs.
- **CORS Middleware**: Enables cross-origin requests.
- **Plotly**: Creates charts for visualization.

### **Frontend (Streamlit)**
- **Streamlit**: Provides an interactive UI.
- **Requests**: Connects to the FastAPI backend.
- **Plotly**: Renders dynamic charts.

---

## 📂 Project Structure
```
📦 QueryVizA_Backend  # Backend Code (FastAPI)
 ┣ 📂 venv            # Virtual Environment
 ┣ 📂 models          # AI Model Handlers
 ┣ 📜 main.py         # FastAPI Main Server
 ┣ 📜 requirements.txt # Backend Dependencies
📦 QueryVizA_Frontend  # Frontend Code (Streamlit)
 ┣ 📜 app.py          # Streamlit UI
 ┣ 📜 requirements.txt # Frontend Dependencies
```

---

## 🔧 Setup Instructions

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-repo/AI-Data-Query-Viz.git
cd AI-Data-Query-Viz
```

### 2️⃣ **Backend Setup (FastAPI)**
```bash
cd QueryVizA_Backend
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3️⃣ **Run Backend Server**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4️⃣ **Frontend Setup (Streamlit)**
```bash
cd ../QueryVizA_Frontend
pip install -r requirements.txt
```

### 5️⃣ **Run Streamlit App**
```bash
streamlit run app.py
```

---

## 🛠 API Endpoints

### **1. Upload File & Process Data**
**Endpoint:** `POST /process-file/`
**Description:** Uploads a CSV file, processes it, and generates AI-powered insights.

**Request:**
```json
{
    "file": "uploaded CSV file",
    "model_name": "llama3-8b-8192",
    "source": "Meta"
}
```

**Response:**
```json
{
    "success": true,
    "recommendations": [{ "chart_type": "bar", "x_axis": "Category", "y_axis": "Sales" }],
    "charts": ["{Plotly JSON data}"]
}
```

---

### **2. Query Data using AI**
**Endpoint:** `POST /process-query/`
**Description:** Generates SQL queries from natural language input and executes them on uploaded data.

**Request:**
```json
{
    "file": "uploaded CSV file",
    "model_name": "gemini-1.5-pro-latest",
    "source": "Gemini",
    "query": "Show top 5 products by sales"
}
```

**Response:**
```json
{
    "success": true,
    "sql": "SELECT * FROM users ORDER BY sales DESC LIMIT 5;",
    "data": [{"Product": "Laptop", "Sales": 50000}]
}
```

---

## 🎯 Features
✅ **Upload CSV and Query Data**
✅ **AI-Powered SQL Query Generation**
✅ **Dynamic Plotly Visualizations**
✅ **FastAPI for Scalable Backend**
✅ **Interactive Streamlit UI**

---

## 📌 To-Do & Future Enhancements
- [ ] Add real-time database support (PostgreSQL/MySQL).
- [ ] Implement multi-user authentication.
- [ ] Improve AI model selection and chart insights.
- [ ] Deploy on AWS/GCP.

---


👨‍💻 **Developed by:** Nikhil Dendeti

