import gradio as gr
import pandas as pd

# grouped: DF amb product, sentiment i dominant_topic
# topic_keywords: dict amb paraules clau per topic
# articles: dict amb articles generats per prod


def diplay_article(product):
    summary = format_summary

demo = gr.Interface(
    fn=display_article, 
)

if __name__ == "__main__":
    demo.launch()