from transformers import pipeline
from datasets import load_dataset

# Load model
classifier = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")

# Load test set (100 samples)
dataset = load_dataset("tweet_eval", "sentiment", split="test[:100]")
texts = dataset['text']
true_labels = dataset['label']  # 0=Negative, 1=Neutral, 2=Positive

# Test and calculate accuracy
correct = 0
for text, true_label in zip(texts, true_labels):
    result = classifier(text)[0]
    pred_label = 2 if result['label'] == 'LABEL_2' else 0 if result['label'] == 'LABEL_0' else 1
    if pred_label == true_label:
        correct += 1
accuracy = correct / len(texts)
print(f"Accuracy: {accuracy:.2f}")