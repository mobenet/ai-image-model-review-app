from scraping.reddit_scraper import RedditReviewScraper
from processing.summarization import summarize_reviews
from processing.classification import classify_reviews
from processing.clustering import cluster_reviews
from generation.article_writer import generate_final_articles
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def main():
    scraper = RedditReviewScraper(
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    suggestions = scraper.get_llm_suggestions("AI image generation models")
    print("suggestions: ", suggestions)
    

    df = scraper.scrape_reviews(
        category_type="AI image generation models",
        custom_subreddits=["StableDiffusion",  
                        "Midjourney",         
                        "aiArt",            
                        "generative",       
                        "ArtificialInteligence",  
                        "Dalle2",            
                        "DeepDream",       
                        "RunwayML"],
        custom_products=["Midjourney",
                        "DALL·E",
                        "Stable Diffusion",
                        "Leonardo AI",
                        "Runway ML",
                        "StarryAI",
                        "NightCafe",
                        "Artbreeder",
                        "Deep Dream",
                        "Dream by Wombo"],
        custom_search_terms=["review",
                        "my experience",
                        "hands-on",
                        "comparison",
                        "tried",
                        "tested",
                        "using",
                        "workflow",
                        "prompting",
                        "best settings",
                        "first impressions",
                        "AI art workflow",
                        "vs Midjourney",
                        "vs DALL·E",
                        "how to use",
                        "what I got",
                        "opinion",
                        "performance",
                        "output quality"]
    )
    products=["Midjourney",
                "DALL·E",
                "Stable Diffusion",
                "Leonardo AI",
                "Runway ML"]
    df = pd.read_csv("./data/ai_image_model_reddit_scraping.csv")
    df = df[df["product"].isin(products)]
    df_subset = (
        df.groupby(["product"], group_keys=False)
        .apply(lambda x: x.sample(n=min(300, len(x)), random_state=42))
        .reset_index(drop=True)
    )

    df = summarize_reviews(df_subset)

    df = classify_reviews(df)

    df = df.dropna(subset=['openai_analysis'])
    df.drop(['category_type', 'search_term', 'title', 'date', 'score', 'comments', 
             'subreddit', 'url', 'Unnamed: 10', 'Unnamed: 11', 'summarization'], 
             axis=1, inplace=True)
    
    grouped, lda_model, vectorizer = cluster_reviews(df)
    generate_final_articles(grouped, lda_model, vectorizer)

if __name__ == "__main__":
    main()
