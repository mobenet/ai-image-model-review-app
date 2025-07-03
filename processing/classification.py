from openai import OpenAI
from dotenv import load_dotenv 
import pandas as pd
import time, os
from tqdm import tqdm

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tqdm.pandas()

def is_processed(x):
    return isinstance(x, str) and x.strip().startswith("{") and len(x.strip()) > 10

def call_openai(text):
    prompt = f"""
        Given the following customer review, classify it as 'Good', 'Neutral', or 'Bad'.
        Also extract 2-3 key points from the review that summarize its main ideas.

        Review:
        \"\"\"{text}\"\"\"

        Return your response as a JSON object like this:
        {{
        "sentiment": "Good",
        "key_points": ["...", "..."]
        }}
        """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error:", e)
        return None


def classify_reviews(df: pd.DataFrame, output_path="./data/reviews_openai_analysis.csv") -> pd.DataFrame:
    df = df.copy()
    if os.path.exists(output_path):
        print(f"Loading progress from: {output_path}")
        df = pd.read_csv(output_path)
    else: 
        df["openai_analysis"] = None

    for i in tqdm(range(len(df))):
        if is_processed(df.loc[i, "openai_analysis"]):
            continue 

        review = str(df.loc[i, "summarization"])
        result = call_openai(review)
        df.loc[i, "openai_analysis"] = result
        time.sleep(1.2)

        if i % 10 == 0 or result is None:
            df.to_csv(output_path, index=False)
    return df