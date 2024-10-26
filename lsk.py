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
from textblob import TextBlob  # Added for sentiment analysis

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the SQLite database for users and documents
def initialize_db():
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    
    # Table to store user information
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, role TEXT)''')
    # Adding a Super Admin account with supreme control
    c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
              ('GreyTempest', 'Likith1206$', 'superadmin'))
    
    # Document and other tables
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, subject TEXT, type TEXT, path TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS study_groups
                 (id INTEGER PRIMARY KEY, group_name TEXT, user_id TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS group_members
                 (id INTEGER PRIMARY KEY, group_id INTEGER, user_id TEXT,
                  FOREIGN KEY (group_id) REFERENCES study_groups (id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS forum_posts
                 (id INTEGER PRIMARY KEY, user_id TEXT, content TEXT, date TEXT)''')
    
    conn.commit()
    conn.close()

# User registration function (accessible only to super admin)
def register_user(username, password, role='user'):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, password, role))
        conn.commit()
        st.success(f"User '{username}' registered successfully!")
    except sqlite3.IntegrityError:
        st.error("Username already exists. Choose a different username.")
    except Exception as e:
        logging.error(f"Error registering user: {e}")
    finally:
        conn.close()

# User authentication function
def authenticate_user(username, password):
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username=? AND password=?', (username, password))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

# Super Admin functionalities: View all users and their passwords
def view_all_users():
    conn = sqlite3.connect('cbse_documents.db')
    c = conn.cursor()
    c.execute('SELECT username, password, role FROM users')
    users = c.fetchall()
    conn.close()
    return users

# Initialize the database at the start
initialize_db()

# Streamlit Application Layout and Functionality
st.title("CBSE Study Application - Secure Access")

# Login and Registration Section
login_section = st.sidebar.container()
login_section.subheader("Login")

username = login_section.text_input("Username")
password = login_section.text_input("Password", type="password")
if login_section.button("Login"):
    role = authenticate_user(username, password)
    if role:
        st.session_state['username'] = username
        st.session_state['role'] = role
        if role == 'superadmin':
            st.success("Welcome, GOD")
            st.subheader("God Mode has been activated")
        else:
            st.success(f"Welcome, {username}!")
    else:
        st.error("Invalid credentials. Please try again.")

# Super Admin Section
if 'role' in st.session_state and st.session_state['role'] == 'superadmin':
    st.sidebar.subheader("Super Admin Panel")
    
    # Register New User
    st.sidebar.write("**Register New User**")
    new_username = st.sidebar.text_input("New Username")
    new_password = st.sidebar.text_input("New Password", type="password")
    user_role = st.sidebar.selectbox("Role", ["user", "admin"])
    if st.sidebar.button("Register User"):
        register_user(new_username, new_password, user_role)
    
    # View all registered users
    st.sidebar.write("**View All Users**")
    if st.sidebar.button("Show Users"):
        users = view_all_users()
        for user in users:
            st.sidebar.write(f"Username: {user[0]}, Password: {user[1]}, Role: {user[2]}")

# Restricted Sections for Authenticated Users
if 'username' in st.session_state:
    st.write(f"Welcome, {st.session_state['username']}! Your role is: {st.session_state['role']}")
    
    # Document upload and analysis section
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Save Documents"):
        for uploaded_file in uploaded_files:
            save_document_to_db("CBSE Subject", uploaded_file.name.split('.')[0], uploaded_file)
            st.success(f"Document {uploaded_file.name} saved successfully!")

    # Load and analyze documents
    subject = st.selectbox("Select Subject", ["Math", "Science", "English"])
    doc_type = st.selectbox("Select Document Type", ["Notes", "Sample Papers"])
    documents = load_documents(subject, doc_type)
    if st.button("Analyze Documents"):
        if documents:
            display_statistical_dashboard(documents)
else:
    st.warning("Please log in to access the application.")
    
# Save document to database
def save_document_to_db(subject, doc_type, path):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        c.execute('INSERT INTO documents (subject, type, path) VALUES (?, ?, ?)', (subject, doc_type, path))
        conn.commit()
    except Exception as e:
        logging.error(f"Error saving document to database: {e}")
    finally:
        conn.close()

# Load documents from the database with caching
@st.cache_data
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
                    pdf_text = reader.pages[0].extract_text()  # Read only first page for efficiency
                    documents.append(pdf_text)
            else:
                with open(path[0], 'r', encoding='utf-8') as file:
                    documents.append(file.read())
        except Exception as e:
            st.error(f"Error reading file {path[0]}: {e}")
            logging.error(f"Error reading file {path[0]}: {e}")
    conn.close()
    return documents

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

# Real-time progress bar for file upload
def upload_progress_bar(total_files):
    progress_bar = st.progress(0)
    for i in range(total_files):
        progress_bar.progress((i + 1) / total_files)

# Improved LDA analysis with user-defined topics
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
        sentiments.append(blob.sentiment.polarity)  # Sentiment polarity (-1 to 1)
    return sentiments

# Use transfer learning models like BERT, GPT, T5 for fine-tuning and generating answers/questions
def load_transfer_learning_models():
    question_model = st.sidebar.text_input("Question Generation Model", "gpt2")
    answer_key_model = st.sidebar.text_input("Answer Key Generation Model", "t5-base")
    
    try:
        question_generator = pipeline("text-generation", model=question_model)
        answer_key_generator = pipeline("text2text-generation", model=answer_key_model)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        logging.error(f"Error loading models: {e}")
        return None, None
    
    return question_generator, answer_key_generator

# Generate statistical analysis dashboard for topic trends and marks distribution
def display_statistical_dashboard(documents):
    lda, feature_names = perform_lda_analysis(documents)
    st.write("LDA Topic Distribution:")
    for topic_idx, topic in enumerate(lda.components_):
        st.write(f"Topic {topic_idx}: " + ", ".join([feature_names[i] for i in topic.argsort()[:-6:-1]]))
    
    tfidf_feature_names, tfidf_scores = perform_tfidf_analysis(documents)
    st.write("TF-IDF Scores:")
    for idx, doc_scores in enumerate(tfidf_scores):
        st.write(f"Document {idx} top features: {', '.join([tfidf_feature_names[i] for i in doc_scores.argsort()[-5:][::-1]])}")

# Function to fetch text from a URL with error handling and validation
def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return ' '.join([p.get_text() for p in soup.find_all('p')])
        else:
            st.error(f"Error fetching URL {url}: Status code {response.status_code}")
            logging.error(f"Error fetching URL {url}: Status code {response.status_code}")
            return ""
    except Exception as e:
        st.error(f"Error fetching URL {url}: {e}")
        logging.error(f"Error fetching URL {url}: {e}")
        return ""

# Function to create and store user progress
def save_user_progress(user_id, subject, score):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        c.execute('INSERT INTO user_progress (user_id, subject, score, date) VALUES (?, ?, ?, ?)', 
                  (user_id, subject, score, str(datetime.now())))
        
        # Check if user gets a reward for the score
        if score >= 80:
            reward = "High Score Badge"
            c.execute('INSERT INTO user_rewards (user_id, reward, date) VALUES (?, ?, ?)', 
                      (user_id, reward, str(datetime.now())))
        
        conn.commit()
    except Exception as e:
        logging.error(f"Error saving user progress: {e}")
    finally:
        conn.close()

# Function to create document summary
def create_document_summary(documents):
    summaries = []
    for document in documents:
        sentences = document.split('. ')
        summary = '. '.join(sentences[:3])  # Summarize by taking first 3 sentences
        summaries.append(summary)
    return summaries

# Function to display user performance analytics
def display_performance_analytics(user_id):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        c.execute('SELECT subject, score, date FROM user_progress WHERE user_id=?', (user_id,))
        progress_records = c.fetchall()
        for record in progress_records:
            st.write(f"Subject: {record[0]}, Score: {record[1]}, Date: {record[2]}")
    except Exception as e:
        logging.error(f"Error fetching user performance: {e}")
    finally:
        conn.close()

# Collaboration Functions
def create_study_group(group_name, user_id):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        c.execute('INSERT INTO study_groups (group_name, user_id) VALUES (?, ?)', (group_name, user_id))
        conn.commit()
        st.success(f"Study group '{group_name}' created successfully!")
    except Exception as e:
        logging.error(f"Error creating study group: {e}")
        st.error(f"Failed to create study group: {e}")
    finally:
        conn.close()

def join_study_group(group_id, user_id):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        c.execute('INSERT INTO group_members (group_id, user_id) VALUES (?, ?)', (group_id, user_id))
        conn.commit()
        st.success(f"You have joined the study group with ID {group_id}!")
    except Exception as e:
        logging.error(f"Error joining study group: {e}")
        st.error(f"Failed to join study group: {e}")
    finally:
        conn.close()

def post_forum_message(user_id, content):
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        c.execute('INSERT INTO forum_posts (user_id, content, date) VALUES (?, ?, ?)', 
                  (user_id, content, str(datetime.now())))
        conn.commit()
        st.success("Your message has been posted to the forum!")
    except Exception as e:
        logging.error(f"Error posting forum message: {e}")
        st.error(f"Failed to post message: {e}")
    finally:
        conn.close()

def load_forum_posts():
    try:
        conn = sqlite3.connect('cbse_documents.db')
        c = conn.cursor()
        c.execute('SELECT user_id, content, date FROM forum_posts')
        posts = c.fetchall()
        for post in posts:
            st.write(f"{post[2]} - {post[0]}: {post[1]}")
    except Exception as e:
        logging.error(f"Error loading forum posts: {e}")
    finally:
        conn.close()

# Initialize the database when the script is run
initialize_db()

# Streamlit application layout and functionality
st.title("CBSE Study Application")

# User registration and authentication section
user_id = st.text_input("Enter your User ID")
if st.button("Login"):
    if user_id:
        st.success("Welcome, " + user_id)
    else:
        st.error("Please enter a valid User ID.")

# Upload document section
uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt"], accept_multiple_files=True)
if st.button("Save Documents"):
    upload_progress_bar(len(uploaded_files))
    for uploaded_file in uploaded_files:
        save_document_to_db("CBSE Subject", uploaded_file.name.split('.')[0], uploaded_file)

# Load documents and analysis section
subject = st.selectbox("Select Subject", ["Math", "Science", "English"])
doc_type = st.selectbox("Select Document Type", ["Notes", "Sample Papers"])
documents = load_documents(subject, doc_type)
if st.button("Analyze Documents"):
    if documents:
        # Perform analysis
        display_statistical_dashboard(documents)

# Create Study Group Section
st.header("Study Groups")
group_name = st.text_input("Enter Study Group Name")
if st.button("Create Study Group"):
    create_study_group(group_name, user_id)

group_id = st.number_input("Enter Study Group ID to Join", min_value=1)
if st.button("Join Study Group"):
    join_study_group(group_id, user_id)

# Discussion Forum Section
st.header("Discussion Forum")
forum_content = st.text_area("Write your message")
if st.button("Post Message"):
    post_forum_message(user_id, forum_content)

st.subheader("Forum Posts")
load_forum_posts()
