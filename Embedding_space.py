import numpy as np
import streamlit as st
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objs as go

# Function to calculate embeddings
def calculate_embeddings(words):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

    inputs = tokenizer(words, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to plot embeddings using Plotly
def plot_embeddings(embeddings, words, method='PCA', dimensions=2,  marker_size=10, marker_color='red', marker_text_size=18):
    st.set_page_config(layout="wide")
    if method == 'PCA':
        reducer = PCA(n_components=dimensions)
    elif method == 't-SNE':
        perplexity = min(30, len(embeddings) - 1)  # Ensure perplexity is less than the number of samples
        reducer = TSNE(n_components=dimensions, perplexity=perplexity, learning_rate=200)
    reduced_embeddings = reducer.fit_transform(embeddings)

    if dimensions == 3:
        trace = go.Scatter3d(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], z=reduced_embeddings[:, 2],
                             mode='markers+text', text=words, textposition='top center',
                             marker=dict(size=marker_size, color=marker_color),
                             textfont=dict(size=marker_text_size))
        layout = go.Layout(title=f'{method} Embedding Space Visualization (3D)', 
                           scene=dict(xaxis_title='Component 1', yaxis_title='Component 2', zaxis_title='Component 3'))
    else:
        trace = go.Scatter(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1],
                           mode='markers+text', text=words, textposition='top center',
                           marker=dict(size=marker_size, color=marker_color),
                           textfont=dict(size=marker_text_size))
        layout = go.Layout(title=f'{method} Embedding Space Visualization (2D)', 
                           xaxis_title='Component 1', yaxis_title='Component 2')    

    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)

# Streamlit interface
st.title("Abstraction layer #1")
st.header("Text Vector Embeddings Visualization")
st.markdown("Visualizing of the text embeddings using BERT-function and PCA / t-SNE methods for visualization.")

st.sidebar.title("Уровни абстракции")
st.sidebar.image("https://imagetolink.com/ib/NuXRj07W6Q.png") 
st.sidebar.markdown("Demo build for the blogpost: \n\n https://t.me/abstraction_layers/11")


# User input for words
user_input = st.text_area("Enter words separated by commas", "кот, собака, время, пространство, time, space, sweet, sour, black, white")
words = [word.strip() for word in user_input.split(',') if word.strip()]


# Load embeddings if words are provided
if words:
    embeddings = calculate_embeddings(words)

    st.write(f" {words[0]}  = ", embeddings[0].reshape(1,-1))

    st.header("t-SNE Visualization")
    col1, col2 = st.columns(2)
    with col1:
        plot_embeddings(embeddings, words, method='t-SNE', dimensions=2)
    with col2:
        plot_embeddings(embeddings, words, method='t-SNE', dimensions=3)
    
    st.header("PCA Visualization")
    col1, col2 = st.columns(2)
    with col1:
        plot_embeddings(embeddings, words, method='PCA', dimensions=2)
    with col2:
        plot_embeddings(embeddings, words, method='PCA', dimensions=3)

else:
    st.write("Please enter some words to visualize their embeddings.")

