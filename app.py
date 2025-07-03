import streamlit as st
import os
import glob

st.title("ImaGen")
st.subheader("AI Image Generation Model Reviews")

ARTICLE_DIR = "outputs"
IMAGE_DIR = "images"

articles = [f for f in os.listdir(ARTICLE_DIR) if f.endswith("_article.txt")]
products = [f.replace("_article.txt", "").replace("_", " ") for f in articles]

selected_product = st.selectbox("Select an image generation model:", products)

if selected_product:
    image_pattern = os.path.join(IMAGE_DIR, f"{selected_product.lower()}*")
    image_files = glob.glob(image_pattern)
    if image_files:
        st.image(image_files[0], use_container_width=True)
    filename = selected_product.replace(" ", "_") + "_article.txt"
    file_path = os.path.join(ARTICLE_DIR, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        article = f.read()

    st.subheader(f"Review Article: {selected_product}")
    st.write(article)
