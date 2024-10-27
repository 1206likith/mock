import os
import sqlite3
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import streamlit as st
from docx import Document
import PyPDF2

# Initialize AI models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
question_generator = pipeline("text-generation", model="facebook/bart-large-mnli")
answer_generator = pipeline("text-generation", model="facebook/bart-large-cnn")
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Advanced Analysis Functions

def classify_questions(content):
    """
    Classify questions by type and estimate difficulty level.
    """
    question_types = []
    for line in content.split('\n'):
        if '?' in line:
            question_type = "MCQ" if "Choose" in line or "Select" in line else "Descriptive"
            question_types.append({"question": line, "type": question_type})
    return question_types

def plot_question_classification(classified_questions):
    """
    Plot distribution of question types.
    """
    df = pd.DataFrame(classified_questions)
    fig = px.histogram(df, x='type', title="Question Type Distribution", color='type')
    st.plotly_chart(fig)

def generate_mock_questions(topic_summary):
    """
    Generate mock questions based on key topics.
    """
    generated_questions = []
    for topic in topic_summary:
        response = question_generator(f"Create a question on: {topic}", max_length=50, num_return_sequences=3)
        generated_questions.extend([res['generated_text'] for res in response])
    return generated_questions

def plot_topics(topics):
    """
    Plot the topics extracted from the document.
    """
    topic_count = [f"Topic {i+1}" for i in range(len(topics))]
    fig = px.bar(x=topic_count, y=[1] * len(topics), title="Extracted Topics", text=topics, labels={'x': "Topics", 'y': "Frequency"})
    st.plotly_chart(fig)

# Main Analysis Function
def analyze_document(content, analysis_type="summary"):
    """
    Analyze a document for specific types of analysis like summarization, topic modeling,
    question classification, mock paper generation, and answer key creation.
    """
    if analysis_type == "summary":
        summary = summarizer(content, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        return {"summary": summary}
    
    elif analysis_type == "topics":
        topics = extract_topics(content)
        return {"topics": topics}
    
    elif analysis_type == "question_classification":
        classified_questions = classify_questions(content)
        return {"classified_questions": classified_questions}
    
    elif analysis_type == "mock_questions":
        topic_summary = extract_topics(content)
        generated_questions = generate_mock_questions(topic_summary)
        return {"generated_questions": generated_questions}
    
    elif analysis_type == "answer_keys":
        questions = [q['question'] for q in classify_questions(content)]
        answer_keys = generate_answer_keys(questions)
        return {"answer_keys": answer_keys}

# Topic Extraction with LDA
def extract_topics(document, num_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([document])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {topic_idx + 1}: {' '.join(topic_words)}")
    return topics

# Read PDF/DOCX
def read_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join(page.extract_text() for page in reader.pages)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    return ""

# Streamlit UI for Document Analysis
def document_analysis_page():
    st.title("CBSE Advanced Document Analysis Tool with Visualizations")
    uploaded_file = st.file_uploader("Upload Document (PDF or DOCX)", type=["pdf", "docx"])
    
    if uploaded_file:
        document_text = read_file(uploaded_file)
        
        st.write("Choose Analysis Type:")
        analysis_type = st.selectbox("Analysis Type", [
            "Summary", 
            "Topic Modeling", 
            "Question Classification", 
            "Mock Paper Generation", 
            "Answer Key Creation"
        ])
        
        if st.button("Analyze Document"):
            result = analyze_document(document_text, analysis_type.lower().replace(" ", "_"))
            
            if analysis_type == "Summary":
                st.subheader("Document Summary")
                st.write(result["summary"])
                
            elif analysis_type == "Topic Modeling":
                st.subheader("Extracted Topics")
                plot_topics(result["topics"])
                    
            elif analysis_type == "Question Classification":
                st.subheader("Classified Questions")
                plot_question_classification(result["classified_questions"])
                for item in result["classified_questions"]:
                    st.write(f"{item['type']}: {item['question']}")
                    
            elif analysis_type == "Mock Paper Generation":
                st.subheader("Generated Mock Questions")
                for question in result["generated_questions"]:
                    st.write(f"- {question}")
                    
            elif analysis_type == "Answer Key Creation":
                st.subheader("Generated Answer Keys")
                for item in result["answer_keys"]:
                    st.write(f"Q: {item['question']}\nA: {item['answer']}")

# Run Streamlit app
if __name__ == "__main__":
    document_analysis_page()