import streamlit as st
from summarizer import summarize_text
import PyPDF2

# Page Config
st.set_page_config(
    page_title="InsightNote",
    page_icon="📘",
    layout="centered"
)

# Title
st.title("📘 InsightNote")
st.subheader("Smart Academic Text Analyzer")

st.write(
    """
    Upload academic content or paste notes to generate
    concise AI-powered summaries using NLP.
    """
)

# Input Type
input_option = st.radio(
    "Choose Input Type:",
    ["Text", "PDF"]
)

text = ""

# TEXT INPUT
if input_option == "Text":

    text = st.text_area(
        "📥 Paste your academic text:",
        height=300
    )

# PDF INPUT
elif input_option == "PDF":

    uploaded_file = st.file_uploader(
        "Upload PDF File",
        type=["pdf"]
    )

    if uploaded_file is not None:

        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        for page in pdf_reader.pages:
            text += page.extract_text()

        st.success("✅ PDF uploaded successfully!")

# Summary Length Slider
summary_length = st.slider(
    "Select Number of Sentences:",
    min_value=1,
    max_value=10,
    value=3
)

# Word Count
if text:
    word_count = len(text.split())
    st.write(f"📊 Word Count: {word_count}")

# Generate Summary
if st.button("Generate Summary"):

    if text.strip() == "":
        st.warning("⚠️ Please provide some text.")

    else:

        with st.spinner("Generating summary..."):

            summary = summarize_text(
                text,
                summary_length
            )

        st.markdown("## 📌 Generated Summary")

        st.success(summary)

# Footer
st.markdown("---")

st.caption(
    "Developed using Python, Streamlit, and NLP "
    "as part of the BYOP AI/ML project."
)
