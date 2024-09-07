import nltk
import torch
from transformers import pipeline
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import re

# Download NLTK stopwords
nltk.download('stopwords')


# Function to process the paragraph and split it into sentences
def process_paragraph(paragraph):
    sentences = paragraph.split(". ")
    processed_sentences = []
    for sentence in sentences:
        # Remove non-alphabetical characters and tokenize the sentence
        cleaned_sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
        processed_sentences.append(cleaned_sentence.split())
    return processed_sentences


# Function to calculate similarity between two sentences
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1 if w not in stopwords]
    sent2 = [w.lower() for w in sent2 if w not in stopwords]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        vector1[all_words.index(w)] += 1

    for w in sent2:
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


# Function to generate a similarity matrix
def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(paragraph, top_n=5):
    stop_words = stopwords.words('english')
    sentences = (process_paragraph(paragraph))

    # Ensure top_n does not exceed the number of sentences
    top_n = min(top_n, len(sentences))

    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summarize_text = []
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentences[i][1]))

    return ". ".join(summarize_text)


# Main program
print("Welcome to AI Summarizer\n")

paragraph = input("Enter the Text\n")
summary = generate_summary(paragraph, top_n=3)
print("Summary:\n", summary)
