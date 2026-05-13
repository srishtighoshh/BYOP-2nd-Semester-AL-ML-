import streamlit as st
import PyPDF2

from summarizer import analyze_text

# PAGE CONFIG
st.set_page_config(
    page_title="InsightNote",
    page_icon="📘",
    layout="centered"
)

# TITLE
st.title("📘 InsightNote")
st.subheader("Smart Academic Text Analyzer")

st.write(
    """
    Analyze academic text, lecture notes, articles,
    and PDFs using NLP-powered summarization.
    """
)

# INPUT TYPE
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

            extracted = page.extract_text()

            if extracted:
                text += extracted

        st.success("✅ PDF uploaded successfully!")

# SUMMARY LENGTH
summary_length = st.slider(
    "Select Number of Summary Sentences:",
    min_value=1,
    max_value=10,
    value=4
)

# WORD COUNT
if text:

    word_count = len(text.split())

    st.write(f"📊 Word Count: {word_count}")

# ANALYZE BUTTON
if st.button("Generate Analysis"):

    if text.strip() == "":

        st.warning("⚠️ Please provide some text.")

    else:

        with st.spinner("Analyzing text..."):

            results = analyze_text(
                text,
                summary_length
            )

        # TITLE
        st.markdown("## 🧠 Generated Title")
        st.info(results["title"])

        # SUMMARY
        st.markdown("## 📌 Summary")

        for sentence in results["summary"]:
            st.write("•", sentence)

        # KEYWORDS
        st.markdown("## 🔑 Keywords")

        st.write(", ".join(results["keywords"]))

        # SENTIMENT
        st.markdown("## 💬 Sentiment Analysis")

        st.success(results["sentiment"])

        # READING TIME
        st.markdown("## ⏱️ Estimated Reading Time")

        st.write(
            f"{results['reading_time']} minutes"
        )

        # TEXT STATS
        st.markdown("## 📊 Text Statistics")

        stats = results["stats"]

        st.write(f"Words: {stats['Words']}")
        st.write(f"Sentences: {stats['Sentences']}")
        st.write(f"Characters: {stats['Characters']}")

        # DOWNLOAD BUTTON
        summary_text = "\n".join(results["summary"])

        st.download_button(
            label="📥 Download Summary",
            data=summary_text,
            file_name="summary.txt",
            mime="text/plain"
        )

# FOOTER
st.markdown("---")

st.caption(
    "Developed using Python, Streamlit, and NLP "
    "for the BYOP AI/ML Project."
)
