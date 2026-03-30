# 📚 InsightNote: Smart Academic Text Analyzer

## 🚀 Overview
InsightNote is a Python-based Natural Language Processing (NLP) tool designed to help students quickly analyze and summarize large amounts of text. It reduces study time by generating concise summaries, extracting key insights, and providing additional analysis like sentiment and reading time.

This project is especially useful during exam preparation when going through lengthy notes becomes time-consuming.


## 🎯 What This Project Does
InsightNote takes a large text input (manual text or PDF) and provides:

- A concise summary of the content  
- Important keywords  
- Bullet-point highlights  
- Sentiment analysis (positive/negative/neutral)  
- Estimated reading time  
- Text statistics (word count, sentence count)  
- Word frequency visualization (graph)  
- Option to save results as a report  


## 🛠️ Technologies Used
- Python  
- NLTK (Natural Language Toolkit)  
- Matplotlib  
- PyPDF2 (optional, for PDF input)  


## 📂 Project Structure
InsightNote/
├── README.md
├── PROJECT REPORT.pdf
└── aimlproject.py

## ⚙️ Setup Instructions

### 1. Install Python
Make sure Python (version 3.8 or above) is installed on your system.

### 2. Clone the Repository
git clone https://github.com/srishtighoshh/InsightNote.git
cd InsightNote

### 3. Install Dependencies
pip install -r requirements.txt

## ▶️ How to Run the Program

Run the following command in terminal:
python main.py

## 🧪 How to Use
1. Choose input type:
   - `1` → Enter text manually  
   - `2` → Provide PDF file path  

2. Enter the number of sentences you want in the summary  

3. The program will generate:
   - Title  
   - Summary  
   - Bullet points  
   - Keywords  
   - Sentiment  
   - Reading time  
   - Text statistics  
   - Word frequency graph  

4. Optionally, choose to save the output as a file  

## 📊 Example Input
Artificial Intelligence is transforming the world rapidly. It is used in healthcare, education, and many industries. However, it also brings challenges like job displacement and ethical concerns.


## 📈 Example Output (Shortened)

- Summary: AI is transforming multiple industries but also introduces challenges.  
- Keywords: artificial, intelligence, industries, challenges  
- Sentiment: Neutral  
- Reading Time: ~0.1 minutes  


## ⚠️ Notes
- PDF feature works only if PyPDF2 is installed  
- First run may take time due to NLTK data download  
- Works best with English text  


## 🚧 Future Improvements
- Web interface (Streamlit)  
- AI-based summarization (Transformers)  
- Multi-language support  
- Voice input  


## 🧠 Learning Outcomes
- Understanding of NLP basics  
- Text preprocessing and tokenization  
- Frequency-based summarization  
- Data visualization  
- Handling real-world coding issues  


## 👩‍💻 Author
BYOP Project Submission  
Srishti 25BAI10321


## ⭐ Acknowledgement
This project was developed as part of a BYOP (Bring Your Own Project) activity to apply programming and problem-solving skills to a real-world scenario.

## 📂 Project Structure
