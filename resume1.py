import streamlit as st
import fitz  # PyMuPDF for better PDF parsing
import google.generativeai as genai
import sqlite3
import tempfile
import os
import json
from sentence_transformers import SentenceTransformer, util

# --- Configure AI Models ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])  # Secure API Key
gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Initialize Database ---
def init_db():
    conn = sqlite3.connect("resumes.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS resumes 
                 (id INTEGER PRIMARY KEY, user TEXT, content TEXT, version TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS job_descriptions 
                 (id INTEGER PRIMARY KEY, user TEXT, content TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- Secure PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF file using PyMuPDF while ensuring proper file handling."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    try:
        with fitz.open(temp_pdf_path) as doc:
            text = "\n".join(page.get_text("text") for page in doc)
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    return text if text.strip() else "No extractable text found."

# --- Resume Structuring into Sections ---
def split_resume_into_sections(text):
    """Splits resume into sections (Experience, Skills, Education)."""
    sections = {"Personal Info": "", "Experience": "", "Skills": "", "Education": "", "Certifications": ""}
    current_section = None

    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("experience") or line.lower().startswith("work experience"):
            current_section = "Experience"
        elif line.lower().startswith("skills"):
            current_section = "Skills"
        elif line.lower().startswith("education"):
            current_section = "Education"
        elif line.lower().startswith("certifications"):
            current_section = "Certifications"
        elif line.lower().startswith("name") or "@" in line:
            current_section = "Personal Info"
        
        if current_section:
            sections[current_section] += line + "\n"

    return sections

# --- AI-Driven Enhancements ---
def get_gemini_response(prompt):
    """Interacts with Gemini AI to enhance or analyze resumes."""
    chat = gemini_model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text

def enhance_resume_section(section_text, section_name):
    """Enhances a specific section of the resume."""
    prompt = f"Improve this '{section_name}' section of a resume:\n{section_text}"
    return get_gemini_response(prompt)

# --- AI Resume & Job Matching ---
def match_job_description(resume_text, job_desc):
    """Calculates AI-powered job match score."""
    resume_embedding = bert_model.encode(resume_text, convert_to_tensor=True)
    job_embedding = bert_model.encode(job_desc, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
    return round(similarity_score * 100, 2)

# --- ATS Compliance Checker ---
def check_ats_compliance(resume_text):
    """Checks ATS friendliness of resume based on section and keyword analysis."""
    required_sections = ["Experience", "Education", "Skills"]
    missing_sections = [section for section in required_sections if section.lower() not in resume_text.lower()]
    return missing_sections

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI Resume Enhancer & Job Matcher", layout="wide")
st.title("🚀 AI-Powered Resume Enhancer & Job Matcher")

# --- Sidebar Navigation ---
page = st.sidebar.radio("Navigation", ["🏆 Resume Enhancement", "🔍 Job Matching", "💬 AI Chat"])

if page == "🏆 Resume Enhancement":
    st.header("📄 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        resume_sections = split_resume_into_sections(resume_text)

        st.subheader("📜 Original Resume (Scrollable A4 Format)")
        st.text_area("Full Resume (Read-Only)", resume_text, height=600, key="original_resume", disabled=True)

        for section, text in resume_sections.items():
            if section != "Personal Info":
                st.subheader(f"✍️ {section} Section")
                st.text_area(f"Original {section}", text, key=f"original_{section}", height=200, disabled=True)
                
                if st.button(f"✨ Improve {section}", key=f"improve_{section}"):
                    improved_text = enhance_resume_section(text, section)
                    st.text_area(f"Enhanced {section}", improved_text, key=f"enhanced_{section}", height=200)

        if st.button("📌 Final Optimized Resume"):
            full_resume = "\n\n".join([f"### {sec}\n{text}" for sec, text in resume_sections.items() if sec != "Personal Info"])
            st.subheader("✅ Optimized & Structured Resume")
            st.text_area("Final Resume", full_resume, height=600)

elif page == "🔍 Job Matching":
    st.header("🎯 Resume & Job Description Matching")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume_matching")
    job_file = st.file_uploader("Upload Job Description (PDF/TXT)", type=["pdf", "txt"], key="job_desc")

    if uploaded_file and job_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        job_desc = extract_text_from_pdf(job_file)

        if st.button("📊 Match Resume with Job"):
            match_score = match_job_description(resume_text, job_desc)
            st.subheader("🔥 Job Match Score")
            st.markdown(f"<h1 style='text-align: center; color: green;'>{match_score}%</h1>", unsafe_allow_html=True)

            if match_score > 75:
                st.success("🎉 Strong match! Apply with confidence!")
            elif match_score > 50:
                st.warning("⚠️ Moderate match! Improve key sections!")
            else:
                st.error("❌ Weak match. Resume needs improvement.")

            st.subheader("📝 Deep Analysis & Improvement Suggestions")
            analysis = get_gemini_response(f"Analyze this resume against the job description and suggest improvements:\nResume:\n{resume_text}\nJob Description:\n{job_desc}")
            st.markdown(analysis)

elif page == "💬 AI Chat":
    st.header("💬 AI Chat for Resume Enhancement & Feedback")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        st.markdown(message)

    user_input = st.text_input("Ask AI about your resume or job match:")

    if user_input:
        ai_response = get_gemini_response(user_input)
        st.session_state.chat_history.append(f"**You:** {user_input}")
        st.session_state.chat_history.append(f"**AI:** {ai_response}")
        st.markdown(f"**AI:** {ai_response}")
