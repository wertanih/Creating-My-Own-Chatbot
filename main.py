# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Read and preprocess the text file
def preprocess(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().lower()

        # Tokenize the text into sentences
        sentences = sent_tokenize(text)

        # Remove punctuation and stop words, and tokenize each sentence into words
        stop_words = set(stopwords.words('english'))
        preprocessed_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
            preprocessed_sentences.append(' '.join(filtered_words))

        return preprocessed_sentences
    except FileNotFoundError:
        st.error("Error: File not found.")
        return []
    except IOError:
        st.error("Error: Unable to open the file.")
        return []

# Compute the similarity between the user's query and each sentence
def get_most_relevant_sentence(query, sentences):
    sentences.append(query)

    # Create TF-IDF vectorizer and transform the sentences into vectors
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)

    # Compute cosine similarity between the query vector and each sentence vector
    similarity_scores = cosine_similarity(sentence_vectors[:-1], sentence_vectors[-1])

    # Get the index of the most similar sentence
    most_relevant_index = similarity_scores.argmax()

    return sentences[most_relevant_index]

# Define the chatbot function
def chatbot(file_path):
    sentences = preprocess(file_path)
    if not sentences:
        return

    st.write("Chatbot: Hello! How can I assist you today?")
    while True:
        user_input = st.text_input("User:")
        if user_input.lower() == 'exit':
            st.write("Chatbot: Goodbye!")
            break

        most_relevant_sentence = get_most_relevant_sentence(user_input, sentences)
        st.write("Chatbot:", most_relevant_sentence)

# Define the main function
def main():
    st.title("Chatbot")
    file_path = st.text_input("Enter the path to your text file:")
    if st.button("Start Chatbot"):
        chatbot(file_path)

if __name__ == '__main__':
    main()
