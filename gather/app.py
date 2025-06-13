import streamlit as st
import json
import datetime
import uuid
import io
from huggingface_hub import HfApi, hf_hub_download

# Load Hugging Face token and repo ID from Streamlit Secrets
hf_token = st.secrets["hf"]["token"]
HF_REPO_ID = st.secrets["hf"]["repo_id"]

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

# Set Streamlit page config
st.set_page_config(page_title="Boteval Response Collector", layout="wide")
st.title("LLM Response Collector")

# Initialize session state for metadata
if "metadata" not in st.session_state:
    st.session_state.metadata = {
        "model_name": "GPT-4o",
        "run_id": f"experiment_{datetime.date.today().isoformat()}",
        "operator": os.getenv("USER", "unknown")
    }

# Metadata input fields
st.subheader("Metadata")

st.session_state.metadata["model_name"] = st.text_input("Model Name", value=st.session_state.metadata["model_name"])
st.session_state.metadata["run_id"] = st.text_input("Run ID", value=st.session_state.metadata["run_id"])
st.session_state.metadata["operator"] = st.text_input("Operator", value=st.session_state.metadata["operator"])

st.markdown("---")

# Initialize session state for responses
if "responses" not in st.session_state:
    st.session_state.responses = {q['id']: "" for q in questions}

# Clear Form button
if st.button("Clear Form"):
    st.session_state.responses = {q['id']: "" for q in questions}
    st.session_state.metadata = {
        "model_name": "GPT-4o",
        "run_id": f"experiment_{datetime.date.today().isoformat()}",
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
        submission = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_name": st.session_state.metadata["model_name"],
            "run_id": st.session_state.metadata["run_id"],
            "operator": st.session_state.metadata["operator"],
            "responses": st.session_state.responses
        }

        # Serialize submission to JSON string
        submission_json = json.dumps(submission, indent=2)

        # Create unique filename
        file_id = str(uuid.uuid4())
        timestamp_safe = submission['timestamp'].replace(":", "-")
        filename = f"gather/submission-{timestamp_safe}-{file_id}.json"

        # Upload submission directly from memory (no local file needed)
        hf_api.upload_file(
            path_or_fileobj=io.BytesIO(submission_json.encode()),
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )

        st.success("Submission uploaded to Hugging Face Hub successfully!")

        # Clear form after submission
        st.session_state.responses = {q['id']: "" for q in questions}
        st.session_state.metadata = {
            "model_name": "GPT-4o",
            "run_id": f"experiment_{datetime.date.today().isoformat()}",
            "operator": os.getenv("USER", "unknown")
        }

