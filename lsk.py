import os
import sqlite3
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from transformers import pipeline
import streamlit as st 
import logging
import numpy as np
import random
from datetime import datetime
import time
import PyPDF2
from fpdf import FPDF
from docx import Document
from textblob import TextBlob
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import torch

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the SQLite database for users, documents, and chat messages
def initialize_db():
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()

    # User and document tables
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, role TEXT)''')
    c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
              ('GreyTempest', 'Likith1206$', 'superadmin'))
    
    # Document, marking schemes, and study groups tables
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, subject TEXT, type TEXT, path TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS marking_schemes
                 (id INTEGER PRIMARY KEY, subject TEXT, path TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS textbooks
                 (id INTEGER PRIMARY KEY, subject TEXT, path TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS study_groups
                 (id INTEGER PRIMARY KEY, group_name TEXT, user_id TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS group_members
                 (id INTEGER PRIMARY KEY, group_id INTEGER, user_id TEXT,
                  FOREIGN KEY (group_id) REFERENCES study_groups (id))''')
    
    # Forum posts and messages table
    c.execute('''CREATE TABLE IF NOT EXISTS forum_posts
                 (id INTEGER PRIMARY KEY, user_id TEXT, content TEXT, date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY, sender_id TEXT, receiver_id TEXT, message TEXT, date TEXT)''')

    # Gamification - User Progress and Points
    c.execute('''CREATE TABLE IF NOT EXISTS user_progress
                 (id INTEGER PRIMARY KEY, user_id TEXT, subject TEXT, score INTEGER, date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS rewards
                 (id INTEGER PRIMARY KEY, user_id TEXT, points INTEGER, badges TEXT)''')

    conn.commit()
    conn.close()

# Initialize the database
initialize_db()

# Authenticate users
def authenticate_user(username, password):
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username=? AND password=?', (username, password))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

# User registration function (super admin only)
def register_user(username, password, role='user'):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, password, role))
        conn.commit()
        st.success(f"User '{username}' registered successfully!")
    except sqlite3.IntegrityError:
        st.error("Username already exists. Choose a different username.")
    finally:
        conn.close()

# Main login page
def login_page():
    st.title("CBSE Study Application - Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        role = authenticate_user(username, password)
        if role:
            st.session_state['username'] = username
            st.session_state['role'] = role
            st.success(f"Welcome, {username}!")
            st.experimental_rerun()  # Refresh the page to access the main app
        else:
            st.error("Invalid credentials. Please try again.")
def view_all_users():
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    c.execute('SELECT username, password, role FROM users')
    users = c.fetchall()
    conn.close()
    return users
# Messaging functionality
def send_message(sender_id, receiver_id, message):
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    c.execute('INSERT INTO messages (sender_id, receiver_id, message, date) VALUES (?, ?, ?, ?)', 
              (sender_id, receiver_id, message, str(datetime.now())))
    conn.commit()
    conn.close()

def view_messages(user_id):
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    c.execute('SELECT sender_id, message, date FROM messages WHERE receiver_id=?', (user_id,))
    messages = c.fetchall()
    conn.close()
    return messages

# Gamification - Assign points and rewards
def assign_points(user_id, points):
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    c.execute('INSERT INTO rewards (user_id, points) VALUES (?, ?)', (user_id, points))
    conn.commit()
    conn.close()

def get_rewards(user_id):
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    c.execute('SELECT points FROM rewards WHERE user_id=?', (user_id,))
    rewards = c.fetchone()
    conn.close()
    return rewards

# Question paper generation using GPT model
def generate_question_paper(subject, template):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    input_text = f"Generate a question paper for {subject} based on the following structure: {template}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=500, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# Answer key generation using GPT model
def generate_answer_key(question_paper):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    input_text = f"Generate an answer key for the following question paper: {question_paper}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=500, num_return_sequences=1)
    answer_key = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer_key

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Pretrained BERT model for semantic similarity

# Predefined CBSE guidelines (this can be loaded from a database or a file)
cbse_guidelines = """
    1. Questions must be divided into sections.
    2. Marks must be evenly distributed as per CBSE structure.
    3. Paper must include MCQs, case-based, short answer, and long answer questions.
    4. Adhere to subject-specific guidelines for subjects like Maths, English, Chemistry, Physics, etc.
"""

# Enhanced overseer_check function
def overseer_check(question_paper, marking_scheme, guidelines=cbse_guidelines):
    """
    Enhanced overseer_check function to verify CBSE compliance by using:
    - Summarization: To condense the question paper and marking scheme.
    - Semantic Similarity: BERT-based model to compare paper and marking scheme with guidelines.
    - Compliance checks: Structural rules to ensure compliance with CBSE guidelines.
    
    Parameters:
    - question_paper: The generated question paper text.
    - marking_scheme: The uploaded marking scheme text.
    - guidelines: CBSE guidelines (as text or database query result).
    
    Returns:
    - compliance_report: A detailed report on compliance and similarity scores.
    """
    
    # Step 1: Summarize the question paper and marking scheme
    summarized_paper = summarizer(question_paper, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    summarized_scheme = summarizer(marking_scheme, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    summarized_guidelines = summarizer(guidelines, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    
    # Step 2: Semantic matching using Sentence-BERT model for comparison
    embeddings_paper = similarity_model.encode(summarized_paper, convert_to_tensor=True)
    embeddings_scheme = similarity_model.encode(summarized_scheme, convert_to_tensor=True)
    embeddings_guidelines = similarity_model.encode(summarized_guidelines, convert_to_tensor=True)
    
    # Calculate semantic similarity
    similarity_paper_guidelines = util.pytorch_cos_sim(embeddings_paper, embeddings_guidelines).item()
    similarity_scheme_guidelines = util.pytorch_cos_sim(embeddings_scheme, embeddings_guidelines).item()

    # Step 3: Check structural compliance (section types, marks distribution)
    compliance_status = check_cbse_compliance(question_paper, marking_scheme)
    
    # Step 4: Generate compliance report
    compliance_report = {
        'summary_question_paper': summarized_paper,
        'summary_marking_scheme': summarized_scheme,
        'similarity_paper_guidelines': round(similarity_paper_guidelines * 100, 2),
        'similarity_scheme_guidelines': round(similarity_scheme_guidelines * 100, 2),
        'cbse_compliance_status': compliance_status
    }
    
    return compliance_report

# Compliance check for structural guidelines
def check_cbse_compliance(question_paper, marking_scheme):
    """
    This function checks whether the question paper and marking scheme adhere to CBSE guidelines.
    It includes checks like section distribution, marks distribution, and question types.
    
    Parameters:
    - question_paper: Text of the question paper.
    - marking_scheme: Text of the marking scheme.
    
    Returns:
    - compliance_status: A status indicating compliance (True/False) and any issues found.
    """
    issues = []
    
    # Sample structural rules for compliance
    if "Section A" not in question_paper or "MCQ" not in question_paper:
        issues.append("Missing Section A or MCQs in the paper.")
    if "case-based" not in question_paper.lower():
        issues.append("Case-based questions missing.")
    
    # Example rule: Ensure marks distribution matches guidelines
    marks_distribution = {"Section A": 20, "Section B": 10, "Section C": 18, "Section D": 20, "Section E": 12}
    total_marks = sum(marks_distribution.values())
    
    # Validate that the paper adheres to CBSE's standard total marks
    if total_marks != 80:
        issues.append(f"Incorrect total marks distribution: {total_marks} instead of 80.")
    
    # Return compliance status and detected issues
    if len(issues) == 0:
        return {"status": True, "message": "The paper complies with CBSE guidelines."}
    else:
        return {"status": False, "issues": issues}

# AI Overseer for CBSE compliance
def upload_training_data(uploaded_files, upload_type):
    """
    This function handles user-uploaded marking schemes, sample papers, or textbooks
    to train models for question-paper generation or answer key creation.
    
    Parameters:
    - uploaded_files: List of uploaded files.
    - upload_type: Type of data being uploaded (marking_scheme, sample_paper, etc.)
    
    Returns:
    - success_message: Indicating successful upload and training start.
    """
    
    for uploaded_file in uploaded_files:
        # Save file to appropriate directory based on upload type
        if upload_type == "marking_scheme":
            directory = "marking_schemes/"
        elif upload_type == "sample_paper":
            directory = "sample_papers/"
        elif upload_type == "textbook":
            directory = "textbooks/"
        else:
            raise ValueError("Invalid upload type")
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Example: Process and train models based on the uploaded data
        st.success(f"File '{uploaded_file.name}' uploaded and model training started.")
        
    return "Upload successful."

# Main application layout
def main_app():
    st.sidebar.title("Navigation")
    st.sidebar.write(f"Logged in as: {st.session_state['username']}")

    # Add a logout button
    if st.sidebar.button("Logout"):
        del st.session_state['username']
        del st.session_state['role']
        st.experimental_rerun()

    # Super Admin Section
    if st.session_state['role'] == 'superadmin':
        st.sidebar.subheader("Super Admin Panel")
        
        new_username = st.sidebar.text_input("New Username")
        new_password = st.sidebar.text_input("New Password", type="password")
        user_role = st.sidebar.selectbox("Role", ["user", "admin"])
        
        if st.sidebar.button("Register User"):
            register_user(new_username, new_password, user_role)
        
        if st.sidebar.button("Show Users"):
            users = view_all_users()
            for user in users:
                st.sidebar.write(f"Username: {user[0]}, Password: {user[1]}, Role: {user[2]}")
        
        # Additional super admin features like document upload
        textbook_file = st.file_uploader("Upload Textbook", type=["pdf", "docx"])
        marking_scheme_file = st.file_uploader("Upload Marking Scheme", type=["pdf", "docx"])
        
        if st.button("Upload Files"):
            if textbook_file:
                save_document_to_db("CBSE Subject", "Textbook", textbook_file)
            if marking_scheme_file:
                save_document_to_db("CBSE Subject", "Marking Scheme", marking_scheme_file)

    # Question paper generation and AI analysis
    st.subheader("Generate Question Paper and Answer Key")
    selected_subject = st.selectbox("Select Subject", ["English", "Math", "Physics", "Chemistry", "Computer Science"])
    paper_template = st.text_area("Enter Paper Structure (e.g. marks distribution, question types)")
    
    if st.button("Generate Question Paper"):
        question_paper = generate_question_paper(selected_subject, paper_template)
        st.text_area("Generated Question Paper", value=question_paper)
        
        if st.button("Generate Answer Key"):
            answer_key = generate_answer_key(question_paper)
            st.text_area("Generated Answer Key", value=answer_key)
    
    # Study group creation
    st.subheader("Create Study Group")
    group_name = st.text_input("Enter Group Name")
    if st.button("Create Study Group"):
        create_study_group(group_name, st.session_state['username'])

    # Uploading and analyzing documents
    subject = st.selectbox("Select Subject", ["Math", "Chemistry", "Physics", "Computer Science", "English"])
    doc_type = st.selectbox("Select Document Type", ["Notes", "Sample Papers"])
    documents = load_documents(subject, doc_type)
    
    if documents:
        st.write(f"Found {len(documents)} documents")
        if st.button("Analyze Documents"):
            display_statistical_dashboard(documents)

    
                
    st.write(f"Welcome to the CBSE Application, {st.session_state['role']}!")

# Question paper generation using GPT model (example function)
def generate_question_paper(subject, template):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    input_text = f"Generate a question paper for {subject} based on the following structure: {template}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=500, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text
# Save document to database
def save_document_to_db(subject, doc_type, uploaded_file):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        
        # Save file content to the appropriate location
        file_path = f"uploads/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Insert the path into the database
        c.execute('INSERT INTO documents (subject, type, path) VALUES (?, ?, ?)', 
                  (subject, doc_type, file_path))
        conn.commit()
        st.success(f"Document '{uploaded_file.name}' uploaded successfully!")
    except Exception as e:
        logging.error(f"Error saving document to database: {e}")
    finally:
        conn.close()

# Load documents from the database with caching
@st.cache_data
def load_documents(subject, doc_type):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        
        # Query the database for documents matching the subject and document type
        c.execute("SELECT path FROM documents WHERE subject = ? AND type = ?", (subject, doc_type))
        documents = c.fetchall()  # This returns a list of tuples
        
        # Extract the file paths from the fetched results
        document_paths = [doc[0] for doc in documents]
        
        conn.close()
        
        return document_paths if document_paths else None  # Return None if no documents are found
    except Exception as e:
        st.error(f"An error occurred while loading documents: {e}")
        return None

# Export document functionality
def export_documents(documents, file_format='PDF'):
    if file_format == 'PDF':
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        for doc in documents:
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, doc)
        pdf_output_path = 'exported_documents.pdf'
        pdf.output(pdf_output_path)
        return pdf_output_path
    elif file_format == 'DOCX':
        doc = Document()
        for document in documents:
            doc.add_paragraph(document)
        doc_output_path = 'exported_documents.docx'
        doc.save(doc_output_path)
        return doc_output_path

# LDA analysis with user-defined topics
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

# KMeans Clustering with user-defined number of clusters
def perform_kmeans_clustering(documents, num_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_

# Sentiment analysis function
def analyze_sentiment(documents):
    sentiments = []
    for document in documents:
        blob = TextBlob(document)
        sentiments.append(blob.sentiment.polarity)
    return sentiments

# Function to create and store user progress
def save_user_progress(user_id, subject, score):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        c.execute('INSERT INTO user_progress (user_id, subject, score, date) VALUES (?, ?, ?, ?)', 
                  (user_id, subject, score, str(datetime.now())))
        conn.commit()
    except Exception as e:
        logging.error(f"Error saving user progress: {e}")
    finally:
        conn.close()

# Create Study Group
def create_study_group(group_name, user_id):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        c.execute('INSERT INTO study_groups (group_name, user_id) VALUES (?, ?)', (group_name, user_id))
        conn.commit()
        st.success(f"Study group '{group_name}' created successfully!")
    except Exception as e:
        logging.error(f"Error creating study group: {e}")
    finally:
        conn.close()

# Add member to a study group
def add_member_to_study_group(group_id, user_id):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        c.execute('INSERT INTO group_members (group_id, user_id) VALUES (?, ?)', (group_id, user_id))
        conn.commit()
        st.success(f"User '{user_id}' added to group {group_id} successfully!")
    except Exception as e:
        logging.error(f"Error adding member to study group: {e}")
    finally:
        conn.close()

# View study group members
def view_study_group_members(group_id):
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    c.execute('SELECT user_id FROM group_members WHERE group_id = ?', (group_id,))
    members = c.fetchall()
    conn.close()
    return [member[0] for member in members]


# Streamlit application logic
if 'username' not in st.session_state:
    login_page()
else:
    main_app()
