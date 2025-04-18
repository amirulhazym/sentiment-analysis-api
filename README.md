Sentiment Analysis API

Overview
This mini-project is a web API that classifies text sentiment as Positive, Negative, or Neutral using the cardiffnlp/twitter-roberta-base-sentiment model from Hugging Face. Built with Streamlit and deployed on Hugging Face Spaces, it provides a user-friendly interface for real-time sentiment analysis, suitable for applications like customer feedback analysis in Malaysia’s fintech and e-commerce sectors.

Live Demo
Will be update here after deployment.

Features

Classifies text sentiment with confidence scores and visual breakdowns (progress bar, bar chart).
Example buttons for quick testing with Positive, Negative, and Neutral inputs.
Responsive Streamlit interface with error handling and model information.
Optimized model loading with @st.cache_resource for efficient deployment.

Installation

Clone the repository:
git clone https://github.com/amirulhazym/sentiment-analysis-api.git
cd sentiment-analysis-api


Create and activate a virtual environment:
python -m venv sa-env
.\sa-env\Scripts\activate  # On Windows


Install dependencies:
pip install -r requirements.txt


Run the app locally:
streamlit run app.py



Usage

Access the app via the live URL or locally.
Enter text in the text area or click example buttons (Positive, Negative, Neutral).
Click "Analyze Sentiment" to view the prediction, confidence score, progress bar, and sentiment breakdown chart.
Expand the "About the Model" section for details on the underlying BERT model.

Model Details

Model: cardiffnlp/twitter-roberta-base-sentiment (RoBERTa-base)
Training Data: ~58M tweets, fine-tuned on TweetEval benchmark
Classes: Negative (LABEL_0), Neutral (LABEL_1), Positive (LABEL_2)
Performance: ~85% accuracy on tweet_eval test set (100 samples)
Limitations: Optimized for short, English, Twitter-like texts; may vary on long or non-English inputs.

Metrics

Accuracy: ~70% on tweet_eval test set (100 samples).
Precision/Recall: Qualitatively aligns with model’s reported performance; full metrics pending further testing due to 1-day constraint.

Relevance to Malaysia/Singapore
This API supports sentiment analysis for customer feedback in fintech (e.g., Grab, CIMB) and e-commerce (e.g., Shopee, Lazada), aligning with Malaysia’s MyDIGITAL initiative and Singapore’s Smart Nation goals. It demonstrates skills in NLP, model deployment, and API development, critical for 20% of AI/ML roles in the region (Jobstreet Report 2024).

Limitations

Limited to single-text input; no batch processing.
English-focused; performance on Bahasa Malaysia is suboptimal (e.g., "Saya suka produk ini!" misclassified as Neutral).
May require fine-tuning for domain-specific applications (e.g., Malaysian social media).

Future Improvements

Fine-tune on Malaysia-specific data (e.g., Malay tweets from brands like AirAsia).
Add support for Bahasa Malaysia to address local language needs.
Implement batch input processing for scalability in high-traffic scenarios.
Enhance with user feedback mechanism for continuous improvement.

Credits

Hugging Face Transformers for the pre-trained model.
Streamlit for the web interface.
PyTorch for the deep learning framework.

Author
Amirulhazym, AI/ML Enthusiast, UTM Electrical & Electronic Engineering Graduate
