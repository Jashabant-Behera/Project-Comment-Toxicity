#!/bin/bash

# Setup script for Comment Toxicity Detection Project
# This script creates the necessary directory structure

echo "Creating project directory structure..."

# Create main directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p notebooks
mkdir -p src
mkdir -p app
mkdir -p results

# Create __init__.py files for Python packages
touch src/__init__.py
touch app/__init__.py

echo "Directory structure created successfully!"
echo ""
echo "Project structure:"
echo "├── data/"
echo "│   ├── raw/              (Place train.csv and test.csv here)"
echo "│   └── processed/"
echo "├── models/               (Trained models will be saved here)"
echo "├── notebooks/            (For EDA and experimentation)"
echo "├── src/                  (Core preprocessing, training, evaluation)"
echo "├── app/                  (Streamlit application)"
echo "└── results/              (Metrics and visualizations)"
echo ""
echo "Next steps:"
echo "1. Place train.csv and test.csv in data/raw/"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Start with EDA in notebooks/"
echo "4. Run preprocessing: python src/preprocess.py"