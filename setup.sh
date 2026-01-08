#!/bin/bash

# Setup script for Comment Toxicity Detection Project

echo "--- Starting Project Setup ---"

# 1. Create Data Directories
echo "Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p results

# 2. Virtual Environment (Optional, user might handle this differently)
# echo "Creating virtual environment..."
# python -m venv venv
# source venv/bin/activate

# 3. Install Dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "--- Setup Complete ---"
echo "Please ensure you have placed 'train.csv' and 'test.csv' in the 'data/raw/' directory."
echo "To run the pipeline:"
echo "1. python src/preprocess.py"
echo "2. python src/train.py"
echo "3. streamlit run app/streamlit_app.py"