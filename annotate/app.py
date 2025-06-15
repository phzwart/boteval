import streamlit as st
import json
import datetime
import uuid
import io
import os
from huggingface_hub import HfApi, hf_hub_download

# Set Streamlit page config - must be first Streamlit command
st.set_page_config(page_title="Boteval Annotation App", layout="wide")

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

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = None

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
                    filename=f"annotate/session-{session_id}.json",
                    repo_type="dataset",
                    token=hf_token
                )
                with open(session_file, "r") as f:
                    session_data = json.load(f)
                st.session_state.session_id = session_id
                st.session_state.annotations = session_data.get("annotations", {})
                st.success("Session loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load session: {str(e)}")
    else:
        if st.button("Create New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.annotations = {}
            st.success(f"New session created! Your session ID is: {st.session_state.session_id}")
            st.info("Please save this session ID to continue your work later.")
            st.rerun()

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

st.title("LLM Question Annotation")

# Display session ID
st.info(f"Current Session ID: {st.session_state.session_id}")

# Get annotator name
annotator = st.text_input("Annotator Name", "")

if not annotator:
    st.warning("Please enter your name to begin.")
    st.stop()

# Topic selection with radio buttons
st.markdown("**Filter by Topic:**")
selected_topic = st.radio(
    "Select Topic to Filter",
    options=["All Topics"] + topics_list,
    index=0,  # Default to "All Topics"
    horizontal=True
)

# Filter questions based on topic selection
if selected_topic == "All Topics":
    questions_to_annotate = questions
else:
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
    st.subheader(f"Question ID: {qid}")
    
    # Display question and answer
    st.markdown("**Question:**")
    st.markdown(q['question'])
    st.markdown("**Answer:**")
    st.markdown(q['answer'])

    # Question quality rating
    st.markdown("**Question Quality Rating:**")
    question_quality = st.radio(
        f"Rate the question quality",
        options=[-1, 0, 1],
        format_func=lambda x: { -1: "Poor (-1)", 0: "Neutral (0)", 1: "Good (+1)" }[x],
        index=1,
        key=f"q_quality_{qid}"
    )

    # Answer quality rating
    st.markdown("**Answer Quality Rating:**")
    answer_quality = st.radio(
        f"Rate the answer quality",
        options=[-1, 0, 1],
        format_func=lambda x: { -1: "Poor (-1)", 0: "Neutral (0)", 1: "Good (+1)" }[x],
        index=1,
        key=f"a_quality_{qid}"
    )

    # Comments field
    comments = st.text_area(
        "Additional Comments",
        value=st.session_state.annotations.get(qid, {}).get("comments", ""),
        key=f"comments_{qid}"
    )

    # Store current question's annotations
    current_annotations = {
        "question_quality": question_quality,
        "answer_quality": answer_quality,
        "comments": comments
    }
    st.session_state.annotations[qid] = current_annotations

    # Submit button for individual question
    if st.button(f"Submit Annotations for {qid}", key=f"submit_{qid}"):
        timestamp = datetime.datetime.now().isoformat().replace(":", "-")
        file_id = str(uuid.uuid4())
        filename = f"annotate/annotation-{timestamp}-{file_id}.json"

        submission = {
            "session_id": st.session_state.session_id,
            "annotator": annotator,
            "timestamp": timestamp,
            "topic": selected_topic,
            "question_id": qid,
            "annotations": {qid: current_annotations}
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
            "annotator": annotator,
            "last_updated": timestamp,
            "annotations": st.session_state.annotations
        }
        session_json = json.dumps(session_data, indent=2)
        hf_api.upload_file(
            path_or_fileobj=io.BytesIO(session_json.encode()),
            path_in_repo=f"annotate/session-{st.session_state.session_id}.json",
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )

        st.success(f"Annotations for {qid} submitted successfully!")

    st.divider()

# Save session button
if st.button("Save Current Session"):
    timestamp = datetime.datetime.now().isoformat()
    session_data = {
        "session_id": st.session_state.session_id,
        "annotator": annotator,
        "last_updated": timestamp,
        "annotations": st.session_state.annotations
    }
    session_json = json.dumps(session_data, indent=2)
    hf_api.upload_file(
        path_or_fileobj=io.BytesIO(session_json.encode()),
        path_in_repo=f"annotate/session-{st.session_state.session_id}.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )
    st.success("Session saved successfully!")

