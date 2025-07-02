from scraping.reddit_scraper import RedditReviewScraper
import os
from dotenv import load_dotenv

load_dotenv()

"""Scraping"""
scraper = RedditReviewScraper(
    reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
    reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

suggestions = scraper.get_llm_suggestions("AI image generation models")
print(suggestions)

subreddits = suggestions["subreddits"]
products = suggestions["products"]
search_terms = suggestions["search_terms"]

df = scraper.scrape_reviews(
    category_type="AI image generation models",
    custom_subreddits=subreddits,
    custom_products=["Midjourney",
                    "DALL·E",
                    "Stable Diffusion",
                    "Leonardo AI",
                    "Runway ML"],
    custom_search_terms=search_terms
)


