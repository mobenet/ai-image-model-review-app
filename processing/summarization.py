from transformers import pipeline, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import re,html

tqdm.pandas()

summarizer = pipeline("text2text-generation", model="google-t5/t5-large")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

def clean(text):
    text = html.unescape(text)  
    text = re.sub(r'https?://\S+', '', text)  
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text) 
    text = re.sub(r'[^\w\s.,!?\'\"]+', '', text)  
    text = re.sub(r'\s+', ' ', text) 
    return text.strip()


def summarize_text(text):
    clean_text = clean(text)
    tokenized = tokenizer.encode(clean_text, truncation=False)
    if len(tokenized) > 512:
        try:
            return summarizer(clean_text, max_new_tokens=300, min_new_tokens=30, no_repeat_ngram_size=3)[0]["generated_text"]
        except Exception as e:
            print(f"Error summarizing: {e}")
        return clean_text
    else:
        return clean_text

def summarize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["summarization"] = df["text"].astype(str).progress_apply(summarize_text)
    return df