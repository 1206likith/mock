import os
import sqlite3
from datetime import datetime
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import PyPDF2
import docx  # Correct import for python-docx

# Initialize AI models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
question_generator = pipeline("text-generation", model="facebook/bart-large-mnli")
answer_generator = pipeline("text-generation", model="facebook/bart-large-cnn")
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# CBSE guidelines
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
c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)", ('GreyTempest', 'Likith1206$', 'superadmin'))
conn.commit()

# Login functionality
def authenticate_user(username, password):
    with sqlite3.connect('cbse_documents.db') as conn:
        c = conn.cursor()
        c.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
        result = c.fetchone()
    return result[0] if result else None

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

# Advanced Analysis Functions
def classify_questions(content):
    question_types = []
    for line in content.split('\n'):
        if '?' in line:
            question_type = "MCQ" if "Choose" in line or "Select" in line else "Descriptive"
            question_types.append({"question": line, "type": question_type})
    return question_types

def generate_mock_questions(topic_summary):
    generated_questions = []
    for topic in topic_summary:
        response = question_generator(f"Create a question on: {topic}", max_length=50, num_return_sequences=3)
        generated_questions.extend([res['generated_text'] for res in response])
    return generated_questions

def generate_answer_keys(questions):
    answer_keys = []
    for question in questions:
        response = answer_generator(question, max_length=150, do_sample=False)
        answer_keys.append({"question": question, "answer": response[0]['generated_text']})
    return answer_keys

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
        # Read DOCX content using python-docx
        document = docx.Document(uploaded_file)
        return " ".join(paragraph.text for paragraph in document.paragraphs)
    return ""

# Analyze Document Function
def analyze_document(document_text, analysis_type):
    result = {}
    if analysis_type == "summary":
        summary = summarizer(document_text, max_length=130, min_length=30, do_sample=False)
        result["summary"] = summary[0]['summary_text']
    elif analysis_type == "topic_modeling":
        result["topics"] = extract_topics(document_text)
    elif analysis_type == "question_classification":
        result["classified_questions"] = classify_questions(document_text)
    elif analysis_type == "mock_paper_generation":
        topics = extract_topics(document_text)
        result["generated_questions"] = generate_mock_questions(topics)
    elif analysis_type == "answer_key_creation":
        questions = classify_questions(document_text)
        result["answer_keys"] = generate_answer_keys([q['question'] for q in questions])
    return result

# Streamlit app structure
def document_analysis_page():
    st.title("CBSE Advanced Document Analysis Tool")

    if 'username' not in st.session_state:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            role = authenticate_user(username, password)
            if role:
                st.session_state['username'] = username
                st.session_state['role'] = role
                st.success(f"Welcome, {username}!")
            else:
                st.error("Invalid credentials. Please try again.")
        return

    st.sidebar.title(f"Welcome, {st.session_state['username']}")
    st.sidebar.subheader("Main Menu")
    page_choice = st.sidebar.selectbox("Choose Page", ["Community Forum", "Gamification", "Advanced Document Analysis"])

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

    elif page_choice == "Advanced Document Analysis":
        st.subheader("Upload Document (PDF or DOCX)")
        uploaded_file = st.file_uploader("Upload Files", type=["pdf", "docx"])

        if uploaded_file:
            document_text = read_file(uploaded_file)
            analysis_type = st.selectbox("Analysis Type", ["Summary", "Topic Modeling", "Question Classification", "Mock Paper Generation", "Answer Key Creation"])

            if st.button("Analyze Document"):
                result = analyze_document(document_text, analysis_type.lower().replace(" ", "_"))

                if analysis_type == "Summary":
                    st.subheader("Document Summary")
                    st.write(result["summary"])
                elif analysis_type == "Topic Modeling":
                    st.subheader("Extracted Topics")
                    for topic in result["topics"]:
                        st.write(topic)
                elif analysis_type == "Question Classification":
                    st.subheader("Classified Questions")
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

if __name__ == "__main__":
    document_analysis_page()
