# Longformer_for_Financial_Sentiment_Analysis

## Model Overview
The `Longformer_for_financial_sentiment_analysis` model is a fine-tuned version of the [Longformer](https://huggingface.co/allenai/longformer-base-4096) by Allen AI, fine-tuned for sentiment analysis of financial text. The fine-tuning process used the [Financial PhraseBank](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10) dataset, a collection of financial news sentences annotated with sentiment. This model is available for use [here](https://huggingface.co/SeanD103/Longformer_for_financial_sentiment_analysis)

## Model Details
- **Model Name**: SeanD103/Longformer_for_financial_sentiment_analysis
- **Model Architecture**: Longformer
- **Pretrained Model**: [Longformer by Allen AI](https://huggingface.co/allenai/longformer-base-4096)
- **Fine-tuned on**: [Financial PhraseBank](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts)
- **Task**: Sentiment Analysis (Positive, Neutral, Negative)

## Model Description
This model is designed to classify the sentiment of financial text, such as news articles, earnings reports, and financial statements. Using the Longformer's ability to handle long documents efficiently, this model can process extended texts up to 4096 tokens, making it suitable for longer financial texts.

## Training Data
The model was fine-tuned on the Financial PhraseBank dataset, which contains 4840 sentences selected from financial news and annotated with their associated sentiment.

## Example Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
model_path = "SeanD103/Longformer_for_financial_sentiment_analysis"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Example input
example_text = "The company's quarterly earnings exceeded expectations, leading to a rise in stock prices."

# Tokenize and get predictions
inputs = tokenizer(example_text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predictions = outputs.logits.argmax(-1)

# Map predictions to labels
sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
sentiment = sentiment_mapping[predictions.item()]
print(f'Input text: {example_text} \nEstimated sentiment: {sentiment}')
```

## Acknowledgments
This model is based on the [Longformer](https://arxiv.org/abs/2004.05150) architecture developed by Allen AI and has been fine-tuned using the [Financial PhraseBank](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts) dataset by Malo et al.


