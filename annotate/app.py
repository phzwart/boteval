import streamlit as st
import json
import datetime
import uuid
import io
import os
from huggingface_hub import HfApi, hf_hub_download

# Load secrets
hf_token = st.secrets["hf"]["token"]
HF_REPO_ID = st.secrets["hf"]["repo_id"]

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

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

hf_api = HfApi(token=hf_token)

# Load questions.json dynamically
questions_file_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="questions.json",
    repo_type="dataset",
    token=hf_token
)

with open(questions_file_path, "r") as f:
    questions = json.load(f)

# Extract all unique topics
topics_set = set()
for q in questions:
    topics = q.get("topic", [])
    if topics:
        topics_set.update(topics)
    else:
        topics_set.add("None")

topics_list = sorted(list(topics_set))

# Initialize Streamlit page
st.set_page_config(page_title="Boteval Annotation App", layout="wide")
st.title("LLM Question Annotation")

# Get annotator name
annotator = st.text_input("Annotator Name", "")

if not annotator:
    st.warning("Please enter your name to begin.")
    st.stop()

# Topic selection
selected_topic = st.selectbox("Select Topic to Annotate", topics_list)

# Filter questions for selected topic
if selected_topic == "None":
    questions_to_annotate = [q for q in questions if not q.get("topic")]
else:
    questions_to_annotate = [
        q for q in questions if selected_topic in q.get("topic", [])
    ]

# Streamlit session state for answers
if "annotations" not in st.session_state:
    st.session_state.annotations = {}

st.divider()

# Annotation loop
for q in questions_to_annotate:
    qid = q["id"]
    st.subheader(f"{qid}: {q['question']}")

    benchmark = st.text_area(
        f"Proposed Answer ({qid})",
        value=st.session_state.annotations.get(qid, {}).get("benchmark", ""),
        key=f"benchmark_{qid}"
    )

    # SAFER RADIO INDEX HANDLING
    options = [-1, 0, 1]
    current_value = st.session_state.annotations.get(qid, {}).get("quality", 1)
    try:
        index = options.index(current_value)
    except ValueError:
        index = options.index(0)

    quality = st.radio(
        f"Question Quality ({qid})",
        options=options,
        format_func=lambda x: { -1: "Bad (-1)", 0: "Neutral (0)", 1: "Good (+1)" }[x],
        index=index,
        key=f"quality_{qid}"
    )

    st.session_state.annotations[qid] = {
        "benchmark": benchmark,
        "quality": quality
    }

st.divider()

# Submit all annotations button
if st.button("Submit Annotations"):
    timestamp = datetime.datetime.now().isoformat().replace(":", "-")
    file_id = str(uuid.uuid4())
    filename = f"annotate/annotation-{timestamp}-{file_id}.json"

    submission = {
        "annotator": annotator,
        "timestamp": timestamp,
        "topic": selected_topic,
        "annotations": st.session_state.annotations
    }

    submission_json = json.dumps(submission, indent=2)
    hf_api.upload_file(
        path_or_fileobj=io.BytesIO(submission_json.encode()),
        path_in_repo=filename,
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )

    st.success("Annotations uploaded successfully!")
    st.session_state.annotations = {}  # reset after submission

