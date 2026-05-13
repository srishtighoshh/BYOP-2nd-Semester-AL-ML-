import streamlit as st
from summarizer import summarize_text

st.set_page_config(page_title="Smart Notes Summarizer")

st.title("🧠 Smart Notes Summarizer")

st.write(
    "Paste your notes below to generate a summary."
)

user_input = st.text_area(
    "Enter your notes:",
    height=250
)

if st.button("Summarize"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        summary = summarize_text(user_input)

        st.subheader("📌 Summary")
        st.success(summary)
