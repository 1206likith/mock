import os
import sqlite3
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from fpdf import FPDF
from docx import Document
import PyPDF2
from transformers import pipeline
import streamlit as st

# Initialize the SQLite database to store document metadata
def initialize_db():
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, subject TEXT, type TEXT, path TEXT)''')
    conn.commit()
    conn.close()

# Save document to database
def save_document_to_db(subject, doc_type, path):
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    c.execute('INSERT INTO documents (subject, type, path) VALUES (?, ?, ?)', (subject, doc_type, path))
    conn.commit()
    conn.close()

# Load documents from the database
def load_documents(subject, doc_type=None):
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    if doc_type:
        c.execute('SELECT path FROM documents WHERE subject=? AND type=?', (subject, doc_type))
    else:
        c.execute('SELECT path FROM documents WHERE subject=?', (subject,))
    paths = c.fetchall()
    documents = []
    for path in paths:
        try:
            if path[0].endswith('.pdf'):
                with open(path[0], 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    pdf_text = ''
                    for page in reader.pages:
                        pdf_text += page.extract_text() + '\n'
                    documents.append(pdf_text)
            else:
                with open(path[0], 'r', encoding='utf-8') as file:
                    documents.append(file.read())
        except Exception as e:
            st.error(f"Error reading file {path[0]}: {e}")
    conn.close()
    return documents

# Improved LDA analysis
def perform_lda_analysis(documents, num_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    return lda, vectorizer.get_feature_names_out()

# TF-IDF Analysis
def perform_tfidf_analysis(documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()
    return feature_names, tfidf_scores

# KMeans Clustering
def perform_kmeans_clustering(documents, num_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_

# Function to fetch text from a URL
def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return ' '.join([p.get_text() for p in soup.find_all('p')])
    except Exception as e:
        st.error(f"Error fetching URL {url}: {e}")
        return ""

# Initialize models
question_generator = pipeline("text-generation", model="gpt2")
overseer_ai = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")  # Replace with actual model

# Streamlit application
def main():
    st.title("CBSE Mock Paper Generator")
    
    # Initialize database
    initialize_db()

    # Subject Selection
    subject = st.selectbox("Select Subject:", ['Chemistry', 'Physics', 'Maths', 'Computer Science', 'English'])

    # Number of Questions
    num_questions = st.number_input("Number of Questions:", min_value=1, max_value=50, value=10)

    # Document Upload Section
    uploaded_files = st.file_uploader("Upload Document(s)", type=["pdf", "txt"], accept_multiple_files=True)

    if st.button("Upload Document(s)"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    doc_type = uploaded_file.name.split('.')[-1]
                    save_document_to_db(subject, doc_type, uploaded_file.name)
                    st.success(f"Uploaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error saving file {uploaded_file.name}: {e}")

    # URL Input
    url = st.text_input("Enter additional resource URL:")
    
    if st.button("Fetch Text from URL"):
        if url:
            fetched_text = fetch_text_from_url(url)
            st.success(f"Fetched text from {url}: {fetched_text[:100]}...")  # Show a snippet

    # Analyze Documents
    if st.button("Analyze Question Papers"):
        documents = load_documents(subject)
        if documents:
            lda, feature_names = perform_lda_analysis(documents)
            st.write("LDA Topics:")
            for topic_idx, topic in enumerate(lda.components_):
                st.write(f"Topic {topic_idx}: " + " ".join([feature_names[i] for i in topic.argsort()[:-6:-1]]))

            tfidf_feature_names, tfidf_scores = perform_tfidf_analysis(documents)
            st.write("TF-IDF Scores:")
            for i, doc in enumerate(tfidf_scores):
                st.write(f"Document {i}:")
                for idx, score in enumerate(doc):
                    if score > 0:
                        st.write(f"  {tfidf_feature_names[idx]}: {score:.4f}")

            cluster_labels = perform_kmeans_clustering(documents)
            st.write("KMeans Clusters:")
            for idx, label in enumerate(cluster_labels):
                st.write(f"Document {idx} belongs to Cluster {label}")

    # Generate Mock Paper
    if st.button("Generate Mock Paper"):
        documents = load_documents(subject)
        mock_questions = create_paper_structure(documents)  # Placeholder call
        st.write("Generated Mock Questions:", mock_questions)

    # Export Options
    if st.button("Export to PDF"):
        paper_structure = create_paper_structure([])  # Placeholder call
        export_to_pdf(paper_structure, subject)
        st.success(f"Exported PDF for {subject}")

    if st.button("Export to Word"):
        paper_structure = create_paper_structure([])  # Placeholder call
        export_to_word(paper_structure, subject)
        st.success(f"Exported Word document for {subject}")

    # Question Prediction
    if st.button("Predict Questions"):
        generated_questions = question_generator(f"Generate {num_questions} questions for {subject}", max_length=200, num_return_sequences=1)
        st.write("Predicted Questions:", generated_questions[0]['generated_text'])

    # View Documents Button
    if st.button("View Uploaded Documents"):
        documents = load_documents(subject)
        for doc in documents:
            st.write("Loaded document:", doc[:100])  # Show a snippet

def create_paper_structure(documents):
    # Placeholder for generating a mock paper structure
    return ["Sample Question 1", "Sample Question 2", "Sample Question 3"]  # Example structure

def export_to_pdf(paper_structure, subject):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Mock Paper - {subject}", ln=True)
    for question in paper_structure:
        pdf.cell(200, 10, question, ln=True)
    pdf_file_path = f"{subject}_mock_paper.pdf"
    pdf.output(pdf_file_path)

def export_to_word(paper_structure, subject):
    doc = Document()
    doc.add_heading(f'Mock Paper - {subject}', 0)
    for question in paper_structure:
        doc.add_paragraph(question)
    word_file_path = f"{subject}_mock_paper.docx"
    doc.save(word_file_path)

if __name__ == "__main__":
    main()
