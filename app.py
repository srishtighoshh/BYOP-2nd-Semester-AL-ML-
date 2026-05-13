import streamlit as st
from summarizer import summarize_text

# Page Configuration
st.set_page_config(
    page_title="InsightNote",
    page_icon="📘",
    layout="centered"
)

# Main Title
st.title("📘 InsightNote")
st.subheader("Smart Academic Text Analyzer")

# Description
st.write(
    """
    InsightNote is an AI-powered academic text analysis tool designed
    to help students quickly summarize lecture notes, articles,
    research content, and study material using Natural Language Processing (NLP).
    """
)

# Feature Section
st.markdown("### ✨ Features")
st.markdown("""
- 📚 Academic text summarization
- ⚡ Fast AI-generated summaries
- 🧠 NLP-based text processing
- 📝 Helpful for revision and quick understanding
""")

# User Input
user_input = st.text_area(
    "📥 Paste your academic text below:",
    height=300,
    placeholder="Enter lecture notes, articles, or study material here..."
)

# Word Count
if user_input:
    word_count = len(user_input.split())
    st.write(f"📊 Word Count: {word_count}")

# Summary Button
if st.button("Generate Summary"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to summarize.")

    else:
        with st.spinner("Analyzing and generating summary..."):

            summary = summarize_text(user_input)

        st.markdown("## 📌 Generated Summary")

        st.success(summary)

# Footer
st.markdown("---")
st.caption(
    "Developed using Python, Streamlit, and NLP techniques "
    "as part of the BYOP AI/ML project."
)
