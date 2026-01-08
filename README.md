# Deep Learning for Comment Toxicity Detection

A real-time comment toxicity detection system using deep learning and Streamlit for automated content moderation.

## Problem Statement

Online platforms face significant challenges with toxic comments including harassment, hate speech, and offensive language. This project develops an automated system to detect and flag toxic comments in real-time, helping moderators maintain healthy online communities.

## Project Overview

- **Domain**: Online Community Management and Content Moderation
- **Techniques**: Deep Learning, NLP, Text Classification
- **Framework**: TensorFlow/Keras, Streamlit
- **Timeline**: 1 week development cycle

## Features

- Text preprocessing pipeline with proper train-test separation
- Deep learning model (LSTM/CNN architecture)
- Real-time single comment prediction
- Bulk CSV upload for batch predictions
- Interactive Streamlit web interface
- Model performance metrics and visualizations

## Project Structure

```
comment-toxicity-detection/
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned datasets
├── models/               # Trained model artifacts
├── notebooks/            # EDA and experimentation
├── src/                  # Core training/evaluation code
├── app/                  # Streamlit application
├── results/              # Metrics and visualizations
├── requirements.txt
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/comment-toxicity-detection.git
cd comment-toxicity-detection
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Place datasets in `data/raw/` folder
   - train.csv
   - test.csv

## Usage

### Training Pipeline

1. **Preprocess data**
```bash
python src/preprocess.py
```

2. **Train model**
```bash
python src/train.py
```

3. **Evaluate model**
```bash
python src/evaluate.py
```

### Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Access the application at `http://localhost:8501`

## Model Performance

Results will be updated after training completion.

- Accuracy: TBD
- Precision: TBD
- Recall: TBD
- F1-Score: TBD

Detailed metrics available in `results/` folder.

## Business Use Cases

1. **Social Media Platforms** - Real-time toxic comment filtering
2. **Online Forums** - Automated content moderation
3. **Content Moderation Services** - Enhanced moderation capabilities
4. **Brand Safety** - Safe advertising environments
5. **E-learning Platforms** - Safe learning environments
6. **News Websites** - User comment moderation

## Technical Stack

- Python 3.8+
- TensorFlow/Keras
- Pandas, NumPy
- NLTK/spaCy
- Scikit-learn
- Streamlit
- Matplotlib, Seaborn

## Project Deliverables

- Trained deep learning model for toxicity detection
- Interactive Streamlit web application
- Comprehensive documentation
- Model evaluation reports
- Source code with modular structure

## Future Enhancements

- Multi-label toxicity classification (severity levels)
- Multilingual support
- Real-time API endpoint
- Model retraining pipeline

## Author

Your Name - [LinkedIn](your-linkedin-url) | [GitHub](your-github-url)

## License

This project is part of a data science capstone program.

## Acknowledgments

- Dataset source: [Link to dataset]
- Project guidelines: Capstone project requirements
- References: NLP and deep learning best practices
