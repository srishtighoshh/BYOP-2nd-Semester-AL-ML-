# -*- coding: utf-8 -*-
"""InsightNote - Smart Academic Text Analyzer"""

import nltk
import string
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# PDF Support
try:
    from PyPDF2 import PdfReader
    pdf_available = True
except ImportError:
    print("⚠️ PyPDF2 not installed. PDF feature disabled.")
    pdf_available = False

# SAFE NLTK DOWNLOADS 
def download_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('stopwords')

    try:
        nltk.data.find('sentiment/vader_lexicon')
    except:
        nltk.download('vader_lexicon')

download_nltk()

# LOAD PDF
def load_pdf(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    except:
        print("⚠️ Error reading PDF. Switching to manual input.")
    return text

# CLEAN TEXT 
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# KEYWORD EXTRACTION
def extract_keywords(text, num_keywords=8):
    stop_words = set(stopwords.words("english"))

    custom_stopwords = {
        "also", "one", "would", "could", "us",
        "like", "get", "make", "many", "much", "is", "was", "were", "has", "have"
    }
    stop_words = stop_words.union(custom_stopwords)

    text = clean_text(text)
    words = word_tokenize(text)

    freq = defaultdict(int)
    for word in words:
        if word.isalpha() and word not in stop_words:
            freq[word] += 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, count in sorted_words[:num_keywords]]

    return keywords  # ✅ fixed (removed extra return)

# SUMMARIZATION
def summarize(text, num_sentences=4):
    stop_words = set(stopwords.words("english"))

    custom_stopwords = {
        "also", "one", "would", "could", "us",
        "like", "get", "make", "many", "much"
    }
    stop_words = stop_words.union(custom_stopwords)

    cleaned_text = clean_text(text)
    words = word_tokenize(cleaned_text)

    freq = defaultdict(int)
    for word in words:
        if word.isalpha() and word not in stop_words:
            freq[word] += 1

    if freq:
        max_freq = max(freq.values())
        for word in freq:
            freq[word] /= max_freq

    sentences = sent_tokenize(text)
    scores = defaultdict(float)

    for sent in sentences:
        word_count = 0
        for word in word_tokenize(sent.lower()):
            if word in freq:
                scores[sent] += freq[word]
                word_count += 1

        if word_count > 0:
            scores[sent] /= word_count

    ranked_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = ranked_sentences[:num_sentences]

    # ✅ FIX: return sentences WITH scores (no output change)
    return sorted(selected, key=lambda x: sentences.index(x[0]))

# TITLE GENERATION
def generate_title(text, keywords):
    sentences = sent_tokenize(text)

    if not sentences:
        return "Generated Summary"

    best_sentence = ""
    max_score = 0

    for sent in sentences:
        words = word_tokenize(sent.lower())  # ✅ FIXED matching
        score = 0
        for word in keywords:
            if word in words:
                score += 1

        if score > max_score:
            max_score = score
            best_sentence = sent

    if best_sentence:
        title = best_sentence.strip()

        if len(title.split()) > 10:
            title = " ".join(title.split()[:10]) + "..."

        return title.title()

    return " ".join(keywords[:3]).title()

# SENTIMENT
def sentiment(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)

    if score['compound'] > 0:
        return "Positive 😊"
    elif score['compound'] < 0:
        return "Negative 😠"
    else:
        return "Neutral 😐"

# READING TIME 
def reading_time(text):
    return round(len(word_tokenize(text)) / 200, 2)

# TEXT STATS 
def text_stats(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    return {
        "Words": len(words),
        "Sentences": len(sentences),
        "Characters": len(text)
    }

# WORD FREQUENCY GRAPH 
def plot_freq(text):
    words = word_tokenize(clean_text(text))
    stop_words = set(stopwords.words("english"))  # ✅ FIXED

    freq = defaultdict(int)
    for w in words:
        if w.isalpha() and w not in stop_words:  # ✅ FIXED
            freq[w] += 1

    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]

    labels = [i[0] for i in top]
    values = [i[1] for i in top]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Most used word frequencies")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# SAVE REPORT 
def save_report(title, summary, keywords):
    with open("summary_report.txt", "w", encoding="utf-8") as f:
        f.write("===== InsightNote Report =====\n\n")
        f.write("TITLE:\n" + title + "\n\n")
        f.write("SUMMARY:\n" + " ".join(summary) + "\n\n")
        f.write("KEYWORDS:\n" + ", ".join(keywords) + "\n")

# MAIN PROGRAM 
print("\n🚀 INSIGHTNOTE: SMART ACADEMIC TEXT ANALYZER 🚀\n")

choice = input("Choose input type:\n1. Manual Text\n2. PDF File\nEnter choice (1/2): ")

if choice == "2" and pdf_available:
    path = input("Enter PDF file path: ")
    text = load_pdf(path)

    if text.strip() == "":
        print("⚠️ PDF empty. Please enter text manually.")
        text = input("\nEnter your text:\n")

else:
    if choice == "2":
        print("⚠️ PDF feature not available. Switching to manual input.")
    text = input("\nEnter your text:\n")

# ✅ FIX: prevent empty crash
if text.strip() == "":
    print("⚠️ Empty input. Please enter valid text.")
    exit()

# Summary length
num = int(input("\nNumber of sentences in summary (1-10): "))
num = max(1, min(10, num))  # ✅ FIX

# Processing
summary_data = summarize(text, num)
summary = [s[0] for s in summary_data]
keywords = extract_keywords(text)
title = generate_title(text, keywords)

# Output
print("\n🧠 GENERATED TITLE:")
print(title)

print("\n📌 SUMMARY:")
print(" ".join(summary))

print("\n🔹 BULLET POINTS:")
for s, score in summary_data:
    print(f"• {s} (Score: {score})")

print("\n🔑 KEYWORDS:")
print(", ".join(keywords))

print("\n💬 SENTIMENT:")
print(sentiment(text))

print("\n⏱️ ESTIMATED READING TIME:")
print(reading_time(text), "minutes")

print("\n📊 TEXT STATISTICS:")
stats = text_stats(text)
for k, v in stats.items():
    print(f"{k}: {v}")

# Graph
plot_freq(text)

# Save option
save = input("\nDo you want to save report? (y/n): ")
if save.lower() == "y":
    save_report(title, summary, keywords)
    print("✅ Report saved as summary_report.txt")

print("\n✅ Analysis Complete!")
