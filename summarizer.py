import nltk
import heapq
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text, num_sentences=3):

    # Clean text
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    formatted_text = re.sub('[^a-zA-Z]', ' ', text)
    formatted_text = formatted_text.lower()

    # Stopwords
    stop_words = set(stopwords.words("english"))

    words = word_tokenize(formatted_text)

    # Word frequencies
    word_frequencies = {}

    for word in words:
        if word not in stop_words:

            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Normalize
    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (
            word_frequencies[word] / maximum_frequency
        )

    # Sentence tokenization
    sentence_list = sent_tokenize(text)

    sentence_scores = {}

    for sentence in sentence_list:

        for word in word_tokenize(sentence.lower()):

            if word in word_frequencies:

                if len(sentence.split(' ')) < 40:

                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_frequencies[word]

                    else:
                        sentence_scores[sentence] += word_frequencies[word]

    # Select top sentences
    summary_sentences = heapq.nlargest(
        num_sentences,
        sentence_scores,
        key=sentence_scores.get
    )

    summary = ' '.join(summary_sentences)

    return summary
