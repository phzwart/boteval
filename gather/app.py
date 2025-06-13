import streamlit as st
import json
import os
import datetime
import uuid
from huggingface_hub import HfApi

# Load config.json
CONFIG_FILE = "config.json"
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

HF_REPO_ID = config["huggingface"]["repo_id"]
DEFAULT_MODEL = config["experiment"]["default_model_name"]
DEFAULT_RUN_ID = config["experiment"]["default_run_id"]

# Load questions.json
QUESTIONS_FILE = "questions.json"
with open(QUESTIONS_FILE, "r") as f:
    questions = json.load(f)

# Load Hugging Face token from secrets
hf_token = st.secrets["hf"]["token"]
hf_api = HfApi(token=hf_token)

# Streamlit page config
st.set_page_config(page_title="PhenixBB LLM Response Collector", layout="wide")
st.title("LLM Response Collector")

# Initialize session state for metadata
if "metadata" not in st.session_state:
    st.session_state.metadata = {
        "model_name": DEFAULT_MODEL,
        "run_id": DEFAULT_RUN_ID,
        "operator": os.getenv("USER", "unknown")
    }

# Metadata input fields
st.subheader("Metadata")

st.session_state.metadata["model_name"] = st.text_input(
    "Model Name", value=st.session_state.metadata["model_name"])
st.session_state.metadata["run_id"] = st.text_input(
    "Run ID", value=st.session_state.metadata["run_id"])
st.session_state.metadata["operator"] = st.text_input(
    "Operator", value=st.session_state.metadata["operator"])

st.markdown("---")

# Initialize session state for responses
if "responses" not in st.session_state:
    st.session_state.responses = {q['id']: "" for q in questions}

# Clear form button
if st.button("Clear Form"):
    st.session_state.responses = {q['id']: "" for q in questions}
    st.session_state.metadata = {
        "model_name": DEFAULT_MODEL,
        "run_id": DEFAULT_RUN_ID,
        "operator": os.getenv("USER", "unknown")
    }
    st.success("Form and metadata cleared!")

# Main form
with st.form("response_form"):
    st.header("Questions")

    for q in questions:
        st.session_state.responses[q['id']] = st.text_area(
            f"**{q['id']}**: {q['question']}",
            height=200,
            value=st.session_state.responses[q['id']]
        )

    submitted = st.form_submit_button("Submit All Responses")

    if submitted:
        # Build submission object
        submission = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_name": st.session_state.metadata["model_name"],
            "run_id": st.session_state.metadata["run_id"],
            "operator": st.session_state.metadata["operator"],
            "responses": st.session_state.responses
        }

        # Serialize to JSON
        submission_json = json.dumps(submission, indent=2)

        # Create unique filename
        file_id = str(uuid.uuid4())
        timestamp_safe = submission['timestamp'].replace(":", "-")
        filename = f"submission-{timestamp_safe}-{file_id}.json"

        # Save temporarily to local file
        tmp_file = f"/tmp/{filename}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(submission_json)

        # Upload to Hugging Face Hub
        hf_api.upload_file(
            path_or_fileobj=tmp_file,
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )

        st.success("Submission uploaded to Hugging Face Hub successfully!")

        # Clear form after submission
        st.session_state.responses = {q['id']: "" for q in questions}
        st.session_state.metadata = {
            "model_name": DEFAULT_MODEL,
            "run_id": DEFAULT_RUN_ID,
            "operator": os.getenv("USER", "unknown")
        }

