import streamlit as st
st.set_page_config(page_title="Boteval Response Collector", layout="wide")

import json
import datetime
import uuid
import io
import os
from huggingface_hub import HfApi, hf_hub_download

# Load Hugging Face token and repo ID from Streamlit Secrets
hf_token = st.secrets["hf"]["token"]
HF_REPO_ID = st.secrets["hf"]["repo_id"]

# Initialize session states
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "metadata" not in st.session_state:
    st.session_state.metadata = {
        "model_name": "GPT-4",
        "run_id": f"experiment_{datetime.date.today().isoformat()}",
        "operator": os.getenv("USER") or os.getenv("USERNAME") or "unknown"
    }

if "responses" not in st.session_state:
    st.session_state.responses = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Authentication function
def authenticate_user(email, password):
    authorized_users = st.secrets.get("authorized_users", {})
    return email in authorized_users and authorized_users[email] == password

# Login form
if not st.session_state.authenticated:
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if authenticate_user(email, password):
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.rerun()
        else:
            st.error("Invalid email or password")
    st.stop()

# Ask for session ID after login
if st.session_state.session_id is None:
    st.title("Session Management")
    session_option = st.radio(
        "Choose session option:",
        ["Start New Session", "Continue Previous Session"]
    )
    
    if session_option == "Continue Previous Session":
        session_id = st.text_input("Enter your session ID:")
        if st.button("Load Session"):
            try:
                # Try to load the session file from HF
                session_file = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=f"gather/session-{session_id}.json",
                    repo_type="dataset",
                    token=hf_token
                )
                with open(session_file, "r") as f:
                    session_data = json.load(f)
                st.session_state.session_id = session_id
                st.session_state.responses = session_data.get("responses", {})
                st.session_state.metadata = session_data.get("metadata", st.session_state.metadata)
                st.success("Session loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load session: {str(e)}")
    else:
        if st.button("Create New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.responses = {}
            st.success(f"New session created! Your session ID is: {st.session_state.session_id}")
            st.info("Please save this session ID to continue your work later.")
            st.rerun()

# Initialize Hugging Face API client
hf_api = HfApi(token=hf_token)

# Download questions.json dynamically from Hugging Face Hub
questions_file_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="questions.json",
    repo_type="dataset",
    token=hf_token
)

with open(questions_file_path, "r") as f:
    questions = json.load(f)

st.title("LLM Response Collector")

# Display session ID
st.info(f"Current Session ID: {st.session_state.session_id}")

# Metadata input fields
st.subheader("Metadata")

st.session_state.metadata["model_name"] = st.text_input("Model Name", value=st.session_state.metadata["model_name"])
st.session_state.metadata["run_id"] = st.text_input("Run ID", value=st.session_state.metadata["run_id"])
st.session_state.metadata["operator"] = st.text_input("Operator", value=st.session_state.metadata["operator"])

st.markdown("---")

# Initialize responses for questions if not exists
if not st.session_state.responses:
    st.session_state.responses = {q['id']: "" for q in questions}

# Clear Form button
if st.button("Clear Form"):
    st.session_state.responses = {q['id']: "" for q in questions}
    st.session_state.metadata = {
        "model_name": "GPT-4",
        "run_id": f"experiment_{datetime.date.today().isoformat()}",
        "operator": os.getenv("USER", "unknown")
    }
    st.success("Form and metadata cleared!")

# Main form
st.header("Questions")

for q in questions:
    qid = q['id']
    st.subheader(f"Question ID: {qid}")
    
    # Create columns for question and copy button
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.markdown(q['question'])
    with col2:
        if st.button("📋", key=f"copy_{qid}", help="Copy question to clipboard"):
            st.write(f'<script>navigator.clipboard.writeText(`{q["question"]}`)</script>', unsafe_allow_html=True)
            st.toast("Question copied to clipboard!")
    
    response = st.text_area(
        "Your Response",
        value=st.session_state.responses.get(qid, ""),
        height=200,
        key=f"response_{qid}"
    )
    
    st.session_state.responses[qid] = response
    
    # Submit button for individual question
    if st.button(f"Submit Response for {qid}", key=f"submit_{qid}"):
        timestamp = datetime.datetime.now().isoformat().replace(":", "-")
        file_id = str(uuid.uuid4())
        filename = f"gather/submission-{timestamp}-{file_id}.json"

        submission = {
            "session_id": st.session_state.session_id,
            "timestamp": timestamp,
            "model_name": st.session_state.metadata["model_name"],
            "run_id": st.session_state.metadata["run_id"],
            "operator": st.session_state.metadata["operator"],
            "question_id": qid,
            "responses": {qid: response}
        }

        submission_json = json.dumps(submission, indent=2)
        hf_api.upload_file(
            path_or_fileobj=io.BytesIO(submission_json.encode()),
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )

        # Save session state
        session_data = {
            "session_id": st.session_state.session_id,
            "metadata": st.session_state.metadata,
            "last_updated": timestamp,
            "responses": st.session_state.responses
        }
        session_json = json.dumps(session_data, indent=2)
        hf_api.upload_file(
            path_or_fileobj=io.BytesIO(session_json.encode()),
            path_in_repo=f"gather/session-{st.session_state.session_id}.json",
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )

        st.success(f"Response for {qid} submitted successfully!")

    st.divider()

# Save session button
if st.button("Save Current Session"):
    timestamp = datetime.datetime.now().isoformat()
    session_data = {
        "session_id": st.session_state.session_id,
        "metadata": st.session_state.metadata,
        "last_updated": timestamp,
        "responses": st.session_state.responses
    }
    session_json = json.dumps(session_data, indent=2)
    hf_api.upload_file(
        path_or_fileobj=io.BytesIO(session_json.encode()),
        path_in_repo=f"gather/session-{st.session_state.session_id}.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )
    st.success("Session saved successfully!")

