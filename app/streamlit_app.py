import streamlit as st
import pandas as pd
import os
from utils import ToxicityPredictor

# Page Config
st.set_page_config(
    page_title="SafeGuard: Toxicity Detection",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    """
    Cached loader for the predictor class to prevent reloading model on every interaction.
    """
    return ToxicityPredictor()

def main():
    st.title("🛡️ SafeGuard")
    st.subheader("AI-Powered Comment Toxicity Detection")
    st.markdown("Enter a comment below to check if it contains toxic content.")

    predictor = get_predictor()

    if predictor.model is None:
        st.error("⚠️ Model not found! Please run the training pipeline first.")
        st.info("Run `python src/train.py` in your terminal.")
        return

    # Tabs for different modes
    tab1, tab2 = st.tabs(["💬 Single Comment", "📁 Batch Processing"])

    # --- TAB 1: SINGLE PREDICTION ---
    with tab1:
        text_input = st.text_area("Type your comment here...", height=150)
        
        if st.button("Analyze Comment", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing semantics..."):
                    score = predictor.predict(text_input)
                
                # Display Results
                st.write("---")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Toxicity Score", f"{score:.1%}")
                
                with col2:
                    if score > 0.5:
                        st.error("🚨 **TOXIC CONTENT DETECTED**")
                        st.write("This comment has been flagged as potential harassment or offensive content.")
                    else:
                        st.success("✅ **CONTENT LOOKS SAFE**")
                        st.write("This comment appears to be within community guidelines.")
                
                # Visual Gauge
                st.progress(score)
            else:
                st.warning("Please enter some text to analyze.")

    # --- TAB 2: BATCH PROCESSING ---
    with tab2:
        st.markdown("Upload a CSV file. It must contain a column named **`comment_text`**.")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            if 'comment_text' in df.columns:
                st.success(f"Loaded {len(df)} comments.")
                
                if st.button("Process All Files"):
                    with st.spinner(f"Analyzing {len(df)} comments..."):
                        # Batch Prediction
                        scores = predictor.predict_batch(df['comment_text'].astype(str).tolist())
                        
                        df['toxicity_score'] = scores
                        df['is_toxic'] = [1 if s > 0.5 else 0 for s in scores] # Binary flag
                        
                        st.dataframe(df[['comment_text', 'toxicity_score', 'is_toxic']].head())
                        
                        # Download Button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name='toxicity_predictions.csv',
                            mime='text/csv'
                        )
            else:
                st.error("Error: CSV must contain a `comment_text` column.")

if __name__ == "__main__":
    main()
