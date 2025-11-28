import os
import re
import tempfile
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

import streamlit as st
import pandas as pd
import numpy as np

from werkzeug.utils import secure_filename

import pdfplumber
import PyPDF2
from docx import Document

from fastembed import TextEmbedding
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv

# ==========================================================
# LOAD ENV
# ==========================================================
load_dotenv()

# ==========================================================
# backend/utils/file_utils.py (adapted for Streamlit files)
# ==========================================================
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc'}


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def secure_temp_save(file_storage):
    """
    Save incoming file (Flask or Streamlit UploadedFile) to a temporary path and return path.
    Caller should clean up file after use.
    """
    # Support both Flask's FileStorage (.filename) and Streamlit's UploadedFile (.name)
    original_name = getattr(file_storage, "filename", None) or getattr(file_storage, "name", None) or "uploaded"
    filename = secure_filename(original_name)
    suffix = os.path.splitext(filename)[1]
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    # Flask-style .save() or Streamlit-style .getbuffer()
    if hasattr(file_storage, "save"):
        file_storage.save(tmp_path)
    else:
        # Assume Streamlit UploadedFile
        data = file_storage.getbuffer()
        with open(tmp_path, "wb") as f:
            f.write(data)

    return tmp_path


# ==========================================================
# backend/utils/skill_mapping.py
# ==========================================================
"""
Simple skill normalization and category mapping.
Extend the mappings as needed.
"""

# normalize common aliases
NORMALIZE_MAP = {
    "aws lambda": "aws",
    "s3": "aws",
    "lambda": "aws",
    "azure functions": "azure",
    "cloud run": "gcp",
    "gcp cloud functions": "gcp"
}

CATEGORY_MAP = {
    "aws": "cloud",
    "azure": "cloud",
    "gcp": "cloud",
    "cloud": "cloud",
    "python": "language",
    "django": "backend",
    "flask": "backend",
    "fastapi": "backend",
    "rest": "api",
    "api": "api",
    "sql": "database",
    "postgres": "database",
    "mysql": "database",
    "docker": "container",
    "kubernetes": "container",
    "machine learning": "ml",
    "ml": "ml",
    "pandas": "ml",
    "numpy": "ml"
}


def normalize_skill(skill: str) -> str:
    if not skill:
        return skill
    s = skill.strip().lower()
    return NORMALIZE_MAP.get(s, s)


def map_skill_to_category(skill: str) -> str:
    s = normalize_skill(skill)
    return CATEGORY_MAP.get(s, s)  # fallback to itself if unknown


# ==========================================================
# backend/models/ranking_result.py
# ==========================================================
@dataclass
class RankingResult:
    """
    Represents a single resume ranking result.
    Holds scoring details for each candidate after
    semantic + skill-based hybrid scoring.
    """
    name: str
    email: Optional[str]
    score: float
    semantic_score: float
    skill_score: float
    skills: List[str]

    def to_dict(self):
        """Convert dataclass object to dictionary for API responses."""
        return asdict(self)


# ==========================================================
# backend/services/skill_extractor.py
# ==========================================================
# Minimal skill list (extend for your domain)
COMMON_SKILLS = [

    # Programming Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "ruby",
    "php", "swift", "kotlin", "r", "scala", "perl", "rust", "dart", "matlab",
    "bash", "shell", "powershell",

    # Web Development
    "html", "css", "bootstrap", "jquery",
    "react", "angular", "vue", "next.js", "nuxt.js", "svelte",
    "node", "express", "django", "flask", "fastapi", "spring boot",
    "laravel", "codeigniter", "wordpress", "drupal",

    # Mobile Development
    "flutter", "react native", "android", "ios", "swiftui", "jetpack compose",

    # Databases
    "sql", "mysql", "postgres", "postgresql", "sqlite",
    "mongodb", "cassandra", "dynamodb", "redis", "elasticsearch",
    "oracle", "mariadb", "snowflake", "bigquery", "couchdb",

    # Cloud Platforms
    "aws", "amazon web services", "lambda", "rds", "ec2", "s3",
    "azure", "azure devops", "azure functions",
    "gcp", "google cloud", "cloud run", "cloud functions",
    "digitalocean", "heroku", "openstack",

    # DevOps & Infrastructure
    "docker", "kubernetes", "helm", "terraform", "ansible", "jenkins",
    "gitlab", "github actions", "cicd", "ci/cd",
    "prometheus", "grafana", "nagios", "elk", "logstash", "kibana",
    "sonarqube", "nexus", "maven", "gradle",

    # Data Science / ML / AI
    "machine learning", "deep learning", "data science", "ai",
    "ml", "dl", "nlp", "computer vision", "ocr",
    "pandas", "numpy", "scikit-learn", "scipy",
    "tensorflow", "keras", "pytorch", "openai", "huggingface",
    "llm", "gpt", "transformers",
    "matplotlib", "seaborn", "plotly",
    "jupyter", "colab",

    # Big Data
    "hadoop", "spark", "pyspark", "hive", "pig", "kafka",
    "airflow", "databricks", "kinesis", "glue",

    # Business Intelligence / Analytics
    "power bi", "tableau", "qlik", "excel", "advanced excel",
    "looker", "microstrategy",

    # AI/ML Ops
    "mlops", "dvc", "mlflow", "kubeflow",

    # Networking & Security
    "tcp/ip", "dns", "vpn", "firewall",
    "cybersecurity", "ethical hacking", "penetration testing",
    "wireshark", "burp suite", "nessus", "splunk",
    "iam", "okta", "oauth", "jwt",

    # Operating Systems
    "linux", "ubuntu", "centos", "windows", "macos",

    # Testing / QA
    "selenium", "pytest", "junit", "testng", "postman",
    "cypress", "jmeter", "rest assured",

    # Project Management / Agile
    "jira", "confluence", "trello", "asana",
    "scrum", "agile", "kanban",

    # Tools & Version Control
    "git", "github", "bitbucket", "svn",
    "vs code", "intellij", "pycharm",
    "eclipse", "postman", "swagger", "figma",

    # Soft Skills / Workflows (optional)
    "communication", "team management", "leadership",
    "presentation", "documentation",

    # Additional Tech Skills
    "api", "rest", "restful", "graphql",
    "microservices", "soa", "distributed systems",
    "sockets", "websockets", "mqtt",

    # AI Cloud Tools
    "vertex ai", "sagemaker", "azure ml",

    # RPA
    "uipath", "automation anywhere", "blue prism",

    # ERP
    "sap", "oracle fusion", "netsuite",

    # Others
    "devsecops", "k6", "bash scripting", "etl", "data warehousing"

    # Incident & Service Management
"incident management",
"problem management",
"change management",
"service management",
"service operations",
"service desk",
"itil",
"itil framework",
"itil v3",
"itil v4",
"service level management",
"service request management",
"sla",
"ola",
"availability management",
"capacity management",
"event management",
"alert management",
"on-call",
"major incident management",
"root cause analysis",
"rca",
"post-incident review",
"p1 incident",
"p2 incident",
"service outage",
"monitoring",
"ticketing",
"jira service desk",
"servicenow",
"bmc remedy",
"solarwinds"

]

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or "").lower()).strip()


def extract_skills_from_text(text: str):
    t = normalize_text(text)
    found = set()

    # exact and boundary-based match
    for skill in COMMON_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, t):
            found.add(skill)

    return list(found)

    # fuzzy-ish checks: individual tokens as fallback
    tokens = set(t.split())
    for token in tokens:
        if token in COMMON_SKILLS:
            found.add(token)
    return list(found)


def extract_emails_from_text(text: str):
    if not text:
        return []
    return EMAIL_REGEX.findall(text)


# ==========================================================
# backend/services/resume_parser.py
# ==========================================================
def extract_text_from_pdf(path):
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text_parts.append(t)
    except Exception:
        # Fallback to PyPDF2
        try:
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for p in reader.pages:
                    t = p.extract_text()
                    if t:
                        text_parts.append(t)
        except Exception as e:
            raise RuntimeError(f"PDF parsing failed: {e}")
    return "\n".join(text_parts).strip()


def extract_text_from_docx(path):
    try:
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        raise RuntimeError(f"DOCX parsing failed: {e}")


def parse_resume(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ('.docx', '.doc'):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type: " + ext)


# ==========================================================
# backend/services/nlp_service.py  (EmbeddingService)
# ==========================================================
class EmbeddingService:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return list(self.model.embed(texts))

    # backward compatibility for scoring_service
    def embed_text(self, texts):
        return self.embed(texts)

    def cosine_sim(self, vec, vec_list):
        vec = np.array(vec)
        vec_list = np.array(vec_list)

        dot = np.dot(vec_list, vec)
        norm_a = np.linalg.norm(vec_list, axis=1)
        norm_b = np.linalg.norm(vec)
        return dot / (norm_a * norm_b + 1e-10)


# ==========================================================
# backend/services/email_service.py
# ==========================================================
class EmailService:
    def __init__(self):
        # Read SMTP settings from environment
        self.smtp_server = os.getenv("SMTP_SERVER", "")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")

        # FROM email (fixed as requested)
        self.sender_email = os.getenv("SENDER_EMAIL", "hrteam@gmail.com")

    def send_candidate_email(self, to_address, candidate_name, job_description, match_score):
        subject = "Congratulations! You Have Been Shortlisted"

        body = f"""
Dear {candidate_name},

We are pleased to inform you that you have been shortlisted for the selection process at "Nav Tech Solution".

Our team will share further details regarding the next steps shortly.
Please keep an eye on your email for updates.

Regards,
HR Team
NAV Tech Solution , Bangalore.
"""

        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = to_address
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.sender_email, to_address, message.as_string())
            return True

        except Exception as e:
            raise Exception(f"Failed to send email: {e}")


# ==========================================================
# backend/services/scoring_service.py
# ==========================================================
class ScoringService:
    def __init__(self, embedding_service, semantic_weight=0.7, skill_weight=0.3):
        """
        embedding_service: instance of EmbeddingService (FastEmbed backend)
        semantic_weight: weight for semantic text similarity scores
        skill_weight: weight for skill match scores
        """
        self.embedding_service = embedding_service
        self.semantic_weight = semantic_weight
        self.skill_weight = skill_weight

    # ----------------------------------------------------------
    # Semantic Similarity
    # ----------------------------------------------------------
    def _semantic_scores(self, job_description, resumes):
        """
        job_description: str
        resumes: list[dict] with 'text'
        """
        # Embed job description → 1 vector
        jd_vec = self.embedding_service.embed_text(job_description)[0]

        # Embed all resume texts
        texts = [r["text"] for r in resumes]
        resume_vecs = self.embedding_service.embed_text(texts)

        # Compute cosine similarity
        sims = self.embedding_service.cosine_sim(jd_vec, resume_vecs)

        # Normalize 0–100
        return (sims * 100).tolist()

    # ----------------------------------------------------------
    # Skill Matching Score
    # ----------------------------------------------------------
    def _skill_match_score(self, jd_skills, resume_skills):
        """
        jd_skills: list[str]
        resume_skills: list[str]
        """
        if not jd_skills:
            return 0.0

        # Normalize and convert skills → categories
          # Normalize skills (exact matching)
        jd_norm = {normalize_skill(s) for s in jd_skills}
        resume_norm = {normalize_skill(s) for s in resume_skills}

        # Exact skill match score
        matched = len(jd_norm.intersection(resume_norm))
        total = max(len(jd_norm), 1)

        return (matched / total) * 100.0


    # ----------------------------------------------------------
    # Final Ranking
    # ----------------------------------------------------------
    def rank(self, job_description, resume_items):
        """
        resume_items: list of dicts:
        {
            "name": str,
            "email": str,
            "text": str,
            "skills": list[str]
        }
        Returns: sorted list by final_score desc
        """

        # Extract skills from JD
        jd_skills = extract_skills_from_text(job_description)

        # Semantic similarity
        semantic_scores = self._semantic_scores(job_description, resume_items)

        results = []

        for i, item in enumerate(resume_items):
            skill_score = self._skill_match_score(jd_skills, item.get("skills", []))
            semantic_score = semantic_scores[i] if i < len(semantic_scores) else 0.0

            final_score = (
                self.semantic_weight * semantic_score +
                self.skill_weight * skill_score
            )

            results.append({
                "name": item.get("name"),
                "email": item.get("email"),
                "score": round(float(final_score), 2),
                "semantic_score": round(float(semantic_score), 2),
                "skill_score": round(float(skill_score), 2),
                "skills": item.get("skills", [])
            })

        # Sort highest → lowest
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ==========================================================
# Initialize services (as in backend)
# ==========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
embedding_service = EmbeddingService()
scoring_service = ScoringService(embedding_service)
email_service = EmailService()


# ==========================================================
# Local helper: process resumes (previously /rank endpoint)
# ==========================================================
def process_resumes(job_description: str, uploaded_files: List) -> List[dict]:
    resume_items = []
    temp_paths = []

    try:
        for f in uploaded_files:
            if f and allowed_file(f.name):
                tmp = secure_temp_save(f)
                temp_paths.append(tmp)

                text = parse_resume(tmp)
                skills = extract_skills_from_text(text)
                emails = extract_emails_from_text(text)

                resume_items.append({
                    "name": os.path.splitext(f.name)[0],
                    "text": text,
                    "skills": skills,
                    "email": emails[0] if emails else None
                })

        results = scoring_service.rank(job_description, resume_items)
        return results

    finally:
        # Cleanup temp files
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass


# ==========================================================
# STREAMLIT UI (frontend/streamlit_app.py adapted)
# ==========================================================
st.set_page_config(page_title="VisionDesk - Resume Screening", layout="wide")
st.title("VisionDesk — Resume Ranking")

# -----------------------------------------------------------
# Input Fields
# -----------------------------------------------------------
job_description = st.text_area(
    "Job description",
    height=250,
    placeholder="Enter the job description here..."
)

uploaded_files = st.file_uploader(
    "Upload resumes (PDF / DOCX)",
    accept_multiple_files=True,
    type=['pdf', 'docx', 'doc']
)

col1, col2 = st.columns([2, 1])

file_count = len(uploaded_files) if uploaded_files else 1

with col1:
    top_n = st.number_input(
        "Top N candidates to show",
        min_value=1,
        max_value=file_count,
        value=min(3, file_count),
        step=1,
        key="dynamic_topn"
    )




# Small helper (kept for compatibility, though not used for HTTP anymore)
def prepare_files_payload(files_list: List) -> List:
    """
    Previously used for requests payload.
    Kept here to respect 'don't remove anything' instruction,
    but not used in local mode.
    """
    out = []
    for f in files_list:
        content_type = getattr(f, "type", "") or "application/octet-stream"
        out.append(('resumes', (f.name, f.getvalue(), content_type)))
    return out

# -----------------------------------------------------------
# Use session state
# -----------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if "top_n" not in st.session_state:
    st.session_state.top_n = 3

# -----------------------------------------------------------
# Main action button
# -----------------------------------------------------------
if st.button("Rank Resumes"):
    if not job_description or not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()

    if not uploaded_files:
        st.error("Please upload at least one resume file.")
        st.stop()

    try:
        with st.spinner("Processing and ranking resumes..."):
            results = process_resumes(job_description, uploaded_files)
            if not results:
                st.warning("No candidates returned. Check uploaded files and job description.")
                st.stop()

        # Save in session state
        st.session_state.results = results
        st.session_state.job_description = job_description
        st.session_state.top_n = top_n

        st.success(f"Successfully ranked {len(results)} candidates! Scroll down to see results.")

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------------------------------------
# SHOW RESULTS IF AVAILABLE
# -----------------------------------------------------------
if st.session_state.results:
    results = st.session_state.results
    job_description = st.session_state.job_description
    top_n = st.session_state.top_n

    # Ensure consistent ordering
    results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    # -----------------------
    # TOP N SUMMARY
    # -----------------------
    st.write(f"### 🏆 Top {int(top_n)} Summary")
    for idx, r in enumerate(results[:int(top_n)], start=1):
        name = r.get("name", "Unknown")
        score = r.get("score", None)
        score_str = f"{score}%" if score is not None else "N/A"
        email = r.get("email", "N/A")
        st.markdown(f"**{idx}. {name}** — Score: `{score_str}` — Email: `{email}`")

# -----------------------
# RANKED CANDIDATES TABLE
# -----------------------
df = pd.DataFrame(results)

# -----------------------
# Columns to show in table (with semantic & skill scores)
# -----------------------
keep_cols = []

if "name" in df.columns:
    keep_cols.append("name")

if "email" in df.columns:
    keep_cols.append("email")

if "score" in df.columns:
    keep_cols.append("score")

if "semantic_score" in df.columns:
    keep_cols.append("semantic_score")

if "skill_score" in df.columns:
    keep_cols.append("skill_score")

if "skills" in df.columns:
    keep_cols.append("skills")

# -----------------------
# Format scores as percentage
# -----------------------
if "semantic_score" in df.columns:
    df["semantic_score"] = df["semantic_score"].apply(lambda v: f"{float(v):.2f}%" if v is not None else "N/A")

if "skill_score" in df.columns:
    df["skill_score"] = df["skill_score"].apply(lambda v: f"{float(v):.2f}%" if v is not None else "N/A")

if "score" in df.columns:
    df["score"] = df["score"].apply(lambda v: f"{float(v):.2f}%" if v is not None else "N/A")

# -----------------------
# Build final table
# -----------------------
if keep_cols:
    df_table = df[keep_cols].copy()

    # Skills as comma-separated
    if "skills" in df_table.columns:
        df_table["skills"] = df_table["skills"].apply(
            lambda s: ", ".join(s) if isinstance(s, (list, tuple)) else (s or "")
        )

    st.write("### ⭐ Ranked Candidates")

    # Row color styling
    def style_table(df_vis: pd.DataFrame):
        def parse(val):
            try:
                if isinstance(val, str) and val.endswith("%"):
                    return float(val.rstrip("%"))
                return float(val)
            except:
                return 0.0

        score_map = {r.get("name", ""): parse(r.get("score", 0)) for r in results}

        def row_style(row):
            nm = row.get("name", "")
            sc = score_map.get(nm, 0)

            if sc >= 70:
                color = "#b3ffb3"
            elif sc >= 50:
                color = "#ffe0b3"
            else:
                color = "#ffcccc"

            return [f"background-color: {color}"] * len(row)

        return df_vis.style.apply(row_style, axis=1)

    st.dataframe(style_table(df_table), use_container_width=True)



    # -------------------------------------------------------
    # SEND EMAIL BUTTON
    # -------------------------------------------------------
    if st.button(f"📧 Send Email to Top {int(top_n)} Candidates"):
        st.write("### 📧 Sending Emails...")

        mail_logs = []
        for r in results[:int(top_n)]:
            email_addr = r.get("email")
            if not email_addr:
                mail_logs.append({"email": None, "status": "no email found"})
                continue

            try:
                email_service.send_candidate_email(
                    email_addr,
                    r.get("name", "Candidate"),
                    job_description,
                    r.get("score", 0),
                )
                mail_logs.append({"email": email_addr, "status": "sent"})
            except Exception as e:
                mail_logs.append({"email": email_addr, "status": f"failed: {e}"})

        st.success("Email send attempt completed. See logs below.")
        st.json(mail_logs)









