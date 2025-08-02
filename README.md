# pyspark-Uber-Trip-Data-
projects analysis using spark
# 🚖 Uber Trip Data Analysis using PySpark

This project analyzes Uber ride data using PySpark. It includes data cleaning, aggregation, and visualization of trip patterns across months, weekdays, and bases.

## 📁 Structure

- `data/`: Data files or download instructions.
- `notebooks/`: Jupyter notebook with visualizations.
- `scripts/`: Python scripts to run via terminal.
- `README.md`: Project explanation.
- `requirements.txt`: Libraries needed.

## 🧪 Features

- Monthly & Daily Trip Analysis
- Trip counts by Base
- Data Aggregation with PySpark
- Optional: Machine Learning extension (KMeans)

## 🚀 How to Run

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python scripts/uber_trip_analysis.py
