import os
import sqlite3
from datetime import datetime
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from docx import Document
import PyPDF2
import pandas as pd

# Initialize AI models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# CBSE guidelines (including computer science)
cbse_guidelines = """
    1. Divide questions into sections.
    2. Distribute marks as per CBSE structure.
    3. Include MCQs, case-based, short answer, and long answer questions.
    4. For Computer Science, ensure practical programming, theory-based, and application-based questions.
"""

# Database setup
conn = sqlite3.connect('cbse_documents.db')
c = conn.cursor()

# Initialize database tables
c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT)")
c.execute("CREATE TABLE IF NOT EXISTS messages (sender_id TEXT, receiver_id TEXT, message TEXT, date TEXT)")
c.execute("CREATE TABLE IF NOT EXISTS rewards (user_id TEXT, points INTEGER)")
conn.commit()

# Pre-populated superadmin user
c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
          ('GreyTempest', 'Likith1206$', 'superadmin'))
conn.commit()

# Login functionality
def authenticate_user(username, password):
    with sqlite3.connect('cbse_documents.db') as conn:
        c = conn.cursor()
        c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
        result = c.fetchone()
    return result[0] if result else None

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
            st.experimental_rerun()
        else:
            st.error("Invalid credentials. Please try again.")

# Forum and Messaging
def send_message(sender_id, receiver_id, message):
    with sqlite3.connect('cbse_documents.db') as conn:
        c = conn.cursor()
        c.execute('INSERT INTO messages (sender_id, receiver_id, message, date) VALUES (?, ?, ?, ?)', 
                  (sender_id, receiver_id, message, str(datetime.now())))
        conn.commit()

def view_messages(user_id):
    with sqlite3.connect('cbse_documents.db') as conn:
        c = conn.cursor()
        c.execute('SELECT sender_id, message, date FROM messages WHERE receiver_id=?', (user_id,))
        return c.fetchall()

# Gamification and Rewards
def assign_points(user_id, points):
    with sqlite3.connect('cbse_documents.db') as conn:
        c = conn.cursor()
        c.execute('INSERT INTO rewards (user_id, points) VALUES (?, ?)', (user_id, points))
        conn.commit()

def get_rewards(user_id):
    with sqlite3.connect('cbse_documents.db') as conn:
        c = conn.cursor()
        c.execute('SELECT points FROM rewards WHERE user_id=?', (user_id,))
        rewards = c.fetchone()
    return rewards[0] if rewards else 0

# Overseer Compliance Check
def overseer_check(question_paper, marking_scheme, guidelines=cbse_guidelines):
    summarized_paper = summarizer(question_paper, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    summarized_scheme = summarizer(marking_scheme, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    summarized_guidelines = summarizer(guidelines, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    
    embeddings_paper = similarity_model.encode(summarized_paper, convert_to_tensor=True)
    embeddings_scheme = similarity_model.encode(summarized_scheme, convert_to_tensor=True)
    embeddings_guidelines = similarity_model.encode(summarized_guidelines, convert_to_tensor=True)
    
    similarity_paper_guidelines = util.pytorch_cos_sim(embeddings_paper, embeddings_guidelines).item()
    similarity_scheme_guidelines = util.pytorch_cos_sim(embeddings_scheme, embeddings_guidelines).item()

    compliance_status = check_cbse_compliance(question_paper, marking_scheme)
    return {
        'summary_question_paper': summarized_paper,
        'summary_marking_scheme': summarized_scheme,
        'similarity_paper_guidelines': round(similarity_paper_guidelines * 100, 2),
        'similarity_scheme_guidelines': round(similarity_scheme_guidelines * 100, 2),
        'cbse_compliance_status': compliance_status
    }

def check_cbse_compliance(question_paper, marking_scheme):
    issues = []
    if "Section A" not in question_paper or "MCQ" not in question_paper:
        issues.append("Missing Section A or MCQs in the paper.")
    if "case-based" not in question_paper.lower():
        issues.append("Case-based questions missing.")
    total_marks = 80
    if sum([20, 10, 18, 20, 12]) != total_marks:
        issues.append(f"Incorrect total marks distribution.")
    return {"status": not issues, "issues": issues}

# Upload Training Data
def upload_training_data(uploaded_files, upload_type):
    for uploaded_file in uploaded_files:
        directory = f"{upload_type}s/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded and model training started.")

# Streamlit app structure
def document_analysis_page():
    st.title("CBSE Study Application")

    # Login page
    if 'username' not in st.session_state:
        login_page()
        return

    # Sidebar options
    st.sidebar.title(f"Welcome, {st.session_state['username']}")
    st.sidebar.subheader("Main Menu")
    page_choice = st.sidebar.selectbox("Choose Page", ["Community Forum", "Gamification", "Overseer Compliance Check", "Upload Training Data"])

    if page_choice == "Community Forum":
        st.subheader("Community Forum")
        user_id = st.session_state['username']
        other_user = st.text_input("Message to (username):")
        message_content = st.text_area("Message Content")
        if st.button("Send Message"):
            send_message(user_id, other_user, message_content)
            st.success("Message sent!")
        st.subheader("Inbox")
        for sender, msg, date in view_messages(user_id):
            st.write(f"From {sender} on {date}: {msg}")

    elif page_choice == "Gamification":
        st.subheader("Gamification - Points and Rewards")
        points_to_assign = st.number_input("Points to assign", min_value=0)
        if st.button("Assign Points"):
            assign_points(st.session_state['username'], points_to_assign)
            st.success(f"{points_to_assign} points assigned.")
        st.write("Your Points:", get_rewards(st.session_state['username']))

    elif page_choice == "Overseer Compliance Check":
        st.subheader("Overseer Compliance Check")
        question_paper = st.text_area("Enter Question Paper Text")
        marking_scheme = st.text_area("Enter Marking Scheme Text")
        if st.button("Check Compliance"):
            report = overseer_check(question_paper, marking_scheme)
            st.write(report)

    elif page_choice == "Upload Training Data":
        st.subheader("Upload Training Data")
        upload_type = st.selectbox("Upload Type", ["marking_scheme", "sample_paper", "textbook"])
        uploaded_files = st.file_uploader("Upload Files", type=["pdf", "docx"], accept_multiple_files=True)
        if st.button("Upload"):
            upload_training_data(uploaded_files, upload_type)
            st.success("Files uploaded successfully.")

if __name__ == "__main__":
    document_analysis_page()