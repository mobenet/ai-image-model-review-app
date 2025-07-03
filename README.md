# ImaGenAI App: An App for AI Image Generation Model Reviews

**ImaGenAI** is a product review aggregator powered by LLMs (Large Language Models). The goal of this project is to help users understand the strengths and weaknesses of popular AI image generation models based on real user feedback collected from Reddit.

## Project Overview

This project was developed as part of the IronHack AI Engineering program. The assignment focuses on building an NLP pipeline to:

* Collect customer reviews from the web
* Classify the reviews by sentiment
* Group them into meaningful topic clusters
* Generate short recommendation articles for each product

## Goals

1. **Classify reviews** as Positive, Neutral, or Negative
2. **Cluster key points** into 4-6 coherent themes using topic modeling
3. **Generate product articles** with GPT-4 summarizing user experiences

## AI Models Used

| Task             | Model                       |
| ---------------- | --------------------------- |
| Summarization    | `google-t5/t5-large`        |
| Sentiment + Keys | `gpt-4` via OpenAI API      |
| Topic Modeling   | TF-IDF + LDA (scikit-learn) |

## Image Generation Models Analyzed

* Midjourney
* DALL·E
* Stable Diffusion
* Leonardo AI
* Runway ML

## Pipeline Overview

1. **Scraping**: Reddit reviews scraped by search terms + subreddit
2. **Preprocessing**: Text cleaned and summarized
3. **Classification**: Sentiment & key points extracted with GPT-4
4. **Clustering**: TF-IDF + LDA per product/sentiment
5. **Generation**: Short article created for each model
6. **Interface**: Streamlit web app to browse results

## Streamlit Web App

To launch the interface:

```bash
streamlit run app.py
```

### Features:

* Select a model and read the AI-generated review summary
* Visual identity and structured layout

## Repository Structure

```
project/
├── scraping/               # Reddit scraping logic
├── processing/             # Summarization, classification, clustering
├── generation/             # Article writing with GPT-4
├── outputs/                # Generated articles (.txt)
├── data/                   # Intermediate data (CSV)
├── app.py                  # Streamlit interface
├── main.py                 # Main pipeline script
└── requirements.txt        # Dependencies
```


## Author

Project created by @mobenet, AI Engineering Student @ IronHack (2025)

## Future Improvements

* Add charts with sentiment distribution
* Deploy app with Hugging Face Spaces or Streamlit Cloud
* Extend to other domains beyond image generation
