import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

import nltk
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import re

nltk.download('stopwords')

def process_paragraph(paragraph):
    # Split the paragraph into sentences
    sentences = paragraph.split(". ")
    return sentences


# Function to clean and tokenize sentences for similarity calculation
def clean_sentence(sentence):
    # sentence tokenizing
    cleaned_sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    return cleaned_sentence.split()


# Function to calculate similarity between two sentences
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in clean_sentence(sent1) if w not in stopwords]
    sent2 = [w.lower() for w in clean_sentence(sent2) if w not in stopwords]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        vector1[all_words.index(w)] += 1

    for w in sent2:
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


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
    sentences = process_paragraph(paragraph)

    top_n = min(top_n, len(sentences))

    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summarize_text = [ranked_sentences[i][1] for i in range(top_n)]

    return ". ".join(summarize_text)


class TextSummarizerApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.title_label = Label(text='Welcome to the Text Summarizer', font_size='20sp', size_hint_y=None, height=40)
        layout.add_widget(self.title_label)

        self.text_input = TextInput(hint_text='Enter your text', multiline=True, size_hint_y=0.7)
        layout.add_widget(self.text_input)

        # Summarize Button
        summarize_button = Button(text='Summarize', size_hint_y=None, height=50)
        summarize_button.bind(on_press=self.summarize_text)
        layout.add_widget(summarize_button)

        self.output_text = TextInput(readonly=True, hint_text='Summary will appear here...', multiline=True,
                                     size_hint_y=0.7)
        layout.add_widget(self.output_text)

        return layout

    def summarize_text(self, instance):
        text = self.text_input.text
        summary = generate_summary(text, top_n=5)  # Call the summary function
        self.output_text.text = summary


# Run the Kivy App
if __name__ == '__main__':
    TextSummarizerApp().run()
