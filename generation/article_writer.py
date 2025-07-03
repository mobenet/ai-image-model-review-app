import os
from openai import OpenAI
from dotenv import load_dotenv 

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_topic_keywords(lda_model, vectorizer, n_words=10):
    topic_keywords = {}
    feature_names = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topic_keywords[idx] = top_words
    return topic_keywords


def generate_prompts(grouped_df, lda_model, vectorizer):
    topic_keywords = get_topic_keywords(lda_model, vectorizer)
    prompts = {}

    providers = grouped_df['product'].unique()

    for provider in providers:
        entry = grouped_df[grouped_df['product'] == provider]
        good_topic = entry[entry['sentiment'] == 'good']['dominant_topic'].values
        bad_topic = entry[entry['sentiment'] == 'bad']['dominant_topic'].values

        good_words = topic_keywords[good_topic[0]] if len(good_topic) else []
        bad_words = topic_keywords[bad_topic[0]] if len(bad_topic) else []

        prompt = f"""
You are a product review writer. Based on user feedback, summarize the main experiences people have with {provider}.

Positive reviews mention topics such as: {', '.join(good_words)}.

Negative reviews focus on: {', '.join(bad_words)}.

Write a short article (2 short paragraphs max) recommending when to use {provider}, what to expect, and potential issues to consider.
""".strip()

        prompts[provider] = prompt

    return prompts



def generate_articles_from_prompts(prompts, output_dir="outputs/", model="gpt-4", max_tokens=250):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = {}
    for provider, prompt in prompts.items():
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful and professional product reviewer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.5
            )
            article = response.choices[0].message.content
            results[provider] = article

            file_path = os.path.join(output_dir, f"{provider.replace(' ', '_')}_article.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(article)

        except Exception as e:
            results[provider] = f"Error: {e}"

    return results


def generate_final_articles(grouped_df, lda_model, vectorizer):
    prompts = generate_prompts(grouped_df, lda_model, vectorizer)
    articles = generate_articles_from_prompts(prompts)
    return articles