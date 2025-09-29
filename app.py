import streamlit as st
from transformers import pipeline
import numpy as np
import torch

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis API",
    page_icon="😀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# App title and description
<<<<<<< HEAD
st.title("Sentiment Analysis API Web Application")
st.write("This API uses a pre-trained RoBERTa-base model to classify text sentiment as positive, negative, or neutral.")
=======
st.title("Sentiment Analysis API")
st.write("This API uses a pre-trained BERT model to classify text sentiment as positive, negative, or neutral.")
>>>>>>> 493dfdf027d4c8d0f8e5465ab0a910941166d775

# Load the sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")

# Get the model
model = load_model()

# Sample text examples
examples = [
    "I absolutely love this new feature! It's amazing.",
    "This product is terrible and doesn't work as advertised.",
    "The weather is just okay today, nothing special."
]

# Create the UI elements
text_input = st.text_area("Enter text to analyze:", height=150, 
                          placeholder="Type or paste your text here...")

# Add example buttons
st.write("Or try one of these examples:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Positive Example"):
        text_input = examples[0]
with col2:
    if st.button("Negative Example"):
        text_input = examples[1]
with col3:
    if st.button("Neutral Example"):
        text_input = examples[2]

# Function to analyze and display sentiment
def analyze_sentiment(text):
    try:
        result = model(text)[0]
        
        # Map labels to user-friendly sentiment names
        sentiment_mapping = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral',
            'LABEL_2': 'Positive'
        }
        
        sentiment = sentiment_mapping[result['label']]
        confidence = result['score']
        
        # Display results with color-coded box
        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
        elif sentiment == "Negative":
            st.error(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
        else:
            st.info(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
            
        # Display confidence as a progress bar
        st.progress(confidence)
        
        # Show detailed sentiment breakdown
        st.subheader("Sentiment Breakdown")
        sentiment_data = {
            'Sentiment': ['Negative', 'Neutral', 'Positive'],
            'Score': [0, 0, 0]  # Default values
        }
        
        # Update the score for the detected sentiment
        if sentiment == "Positive":
            sentiment_data['Score'][2] = confidence
        elif sentiment == "Negative":
            sentiment_data['Score'][0] = confidence
        else:
            sentiment_data['Score'][1] = confidence
            
        # Display as a horizontal bar chart
        st.bar_chart(sentiment_data, x='Sentiment', y='Score')
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Process the text when the analyze button is clicked
if st.button("Analyze Sentiment") and text_input:
    with st.spinner("Analyzing sentiment..."):
        analyze_sentiment(text_input)
elif text_input:
    st.info("Click 'Analyze Sentiment' to process the text.")
else:
    st.info("Please enter some text to analyze.")

def analyze_sentiment(text):
    try:
        if len(text.split()) > 512:
            st.error("Input too long (max 512 words). Please shorten the text.")
            return
        result = model(text)[0]
        sentiment_mapping = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral',
            'LABEL_2': 'Positive'
        }
        sentiment = sentiment_mapping[result['label']]
        confidence = result['score']
        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
        elif sentiment == "Negative":
            st.error(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
        else:
            st.info(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
        st.progress(confidence)
        st.subheader("Sentiment Breakdown")
        sentiment_data = {
            'Sentiment': ['Negative', 'Neutral', 'Positive'],
            'Score': [0, 0, 0]
        }
        if sentiment == "Positive":
            sentiment_data['Score'][2] = confidence
        elif sentiment == "Negative":
            sentiment_data['Score'][0] = confidence
        else:
            sentiment_data['Score'][1] = confidence
        st.bar_chart(sentiment_data, x='Sentiment', y='Score')
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}. Please try again or use shorter text.")

# Add information about the model
with st.expander("About the Model"):
    st.write("""
    This application uses the `cardiffnlp/twitter-roberta-base-sentiment` model from Hugging Face.
    
    The model is a RoBERTa-base model trained on ~58M tweets and fine-tuned for sentiment analysis 
    with the TweetEval benchmark. It classifies text into three sentiment categories:
    
    - Negative (LABEL_0)
    - Neutral (LABEL_1)
    - Positive (LABEL_2)
    
    Source: [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
    """)

with st.expander("Model Performance"):
    st.write("Tested on 100 samples from `tweet_eval` dataset.")
    if st.button("Show Test Accuracy"):
        from datasets import load_dataset
        dataset = load_dataset("tweet_eval", "sentiment", split="test[:100]")
        correct = sum(1 for text, label in zip(dataset['text'], dataset['label']) if (2 if model(text)[0]['label'] == 'LABEL_2' else 0 if model(text)[0]['label'] == 'LABEL_0' else 1) == label)
        st.write(f"Accuracy: {correct/100:.2f}")
        
# Footer
st.markdown("---")
<<<<<<< HEAD
st.markdown("Mini Project 3: Sentiment Analysis API Web Application with RoBERTa and Streamlit by Amirulhazym")

# Add to your footer section
st.markdown("""
<div class="footer">
    © <br>Menggunakan data dari: 
    Barbieri, F. et al. (2020). TweetEval: Unified Benchmark for Tweet Classification. EMNLP.
</div>
""", unsafe_allow_html=True)
=======
st.markdown("Mini Project 3: Sentiment Analysis API")
>>>>>>> 493dfdf027d4c8d0f8e5465ab0a910941166d775
