import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
"""
Create a word cloud based on the frequency of terms in the prompt texts. 
"""
def prompt_word_cloud(graph_dir: str, dataset_path: str):
    df = pd.read_csv(dataset_path)

    # Combine all prompt text into one string
    text = " ".join(df["prompt"].astype(str))

    # Generate word cloud
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        max_words=200
    ).generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(graph_dir + "/wordcloud.pdf")