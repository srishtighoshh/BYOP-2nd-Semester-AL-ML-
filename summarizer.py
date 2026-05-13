import nltk
import heapq
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text):

    # Remove extra spaces and references
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Format text
    formatted_text = re.sub('[^a-zA-Z]', ' ', text)
    formatted_text = formatted_text.lower()

    # Tokenize words
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(formatted_text)

    # Calculate word frequencies
    word_frequencies = {}

    for word in words:
        if word not in stop_words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Normalize frequencies
    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (
            word_frequencies[word] / maximum_frequency
        )

    # Tokenize sentences
    sentence_list = sent_tokenize(text)

    sentence_scores = {}

    # Score sentences
    for sentence in sentence_list:
        for word in word_tokenize(sentence.lower()):

            if word in word_frequencies:

                if len(sentence.split(' ')) < 30:

                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_frequencies[word]

                    else:
                        sentence_scores[sentence] += word_frequencies[word]

    # Select top 3 sentences
    summary_sentences = heapq.nlargest(
        3,
        sentence_scores,
        key=sentence_scores.get
    )

    # Join sentences
    summary = ' '.join(summary_sentences)

    return summary
