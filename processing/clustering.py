from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import json

def cluster_reviews(df: pd.DataFrame,n_topics: int = 3):
    df['openai_analysis'] = df['openai_analysis'].apply(json.loads)
    df_analisis = pd.json_normalize(df['openai_analysis'])
    df = df.join(df_analisis)
    df.drop(['openai_analysis'], axis=1, inplace=True)

    df['sentiment'] = df['sentiment'].str.lower()
    df = df[df['sentiment'].isin(['good', 'bad'])]

    df['key_points'] = df['key_points'].apply(lambda kp: ' '.join(kp))
    grouped = df.groupby(['product', 'sentiment'])['key_points'].apply(lambda texts: ' '.join(texts)).reset_index()

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(grouped['key_points'])

    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(tfidf_matrix)
    topic_distribution = lda_model.transform(tfidf_matrix)
    grouped['dominant_topic'] = topic_distribution.argmax(axis=1)

    return grouped, lda_model, vectorizer