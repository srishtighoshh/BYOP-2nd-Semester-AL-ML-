from main import (
    summarize,
    extract_keywords,
    generate_title,
    sentiment,
    reading_time,
    text_stats
)

def analyze_text(text, num_sentences):

    summary_data = summarize(text, num_sentences)

    summary = [s[0] for s in summary_data]

    keywords = extract_keywords(text)

    title = generate_title(text, keywords)

    senti = sentiment(text)

    read_time = reading_time(text)

    stats = text_stats(text)

    return {
        "title": title,
        "summary": summary,
        "keywords": keywords,
        "sentiment": senti,
        "reading_time": read_time,
        "stats": stats
    }
