import streamlit as st
import json
import io
from huggingface_hub import HfApi, hf_hub_download
import glob
import os

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

# Initialize Hugging Face API client
hf_api = HfApi(token=hf_token)

# Set page config
st.set_page_config(page_title="Boteval Comparison Tool", layout="wide")
st.title("LLM Response Comparison Tool")

# Function to load questions
@st.cache_data(ttl=60)
def load_questions():
    questions_file_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="questions.json",
        repo_type="dataset",
        token=hf_token
    )
    with open(questions_file_path, "r") as f:
        return json.load(f)

# Function to load session metadata
@st.cache_data(ttl=60)
def load_session_metadata():
    sessions = []
    # List all files in the gather directory
    files = hf_api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
    session_files = [f for f in files if f.startswith("gather/session-")]
    
    for file in session_files:
        try:
            file_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=file,
                repo_type="dataset",
                token=hf_token
            )
            with open(file_path, "r") as f:
                session_data = json.load(f)
                sessions.append({
                    "session_id": session_data["session_id"],
                    "metadata": session_data["metadata"],
                    "last_updated": session_data["last_updated"],
                    "filename": file
                })
        except Exception as e:
            st.warning(f"Could not load session file {file}: {str(e)}")
    
    return sessions

# Function to load responses from selected sessions
@st.cache_data(ttl=60)
def load_responses_from_sessions(selected_sessions):
    responses = []
    for session in selected_sessions:
        try:
            file_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=session["filename"],
                repo_type="dataset",
                token=hf_token
            )
            with open(file_path, "r") as f:
                session_data = json.load(f)
                responses.append({
                    "session_id": session_data["session_id"],
                    "metadata": session_data["metadata"],
                    "responses": session_data["responses"]
                })
        except Exception as e:
            st.warning(f"Could not load session {session['session_id']}: {str(e)}")
    
    return responses

# Load questions and session metadata
questions = load_questions()
sessions = load_session_metadata()

# Display session selection
st.header("Select Sessions to Compare")

# Create a list of session options with metadata
session_options = {
    f"{s['metadata']['model_name']} - {s['metadata']['run_id']} - {s['last_updated']}": s 
    for s in sessions
}

# Allow multiple session selection
selected_session_keys = st.multiselect(
    "Choose sessions to compare:",
    options=list(session_options.keys()),
    format_func=lambda x: x
)

# Load responses from selected sessions
selected_sessions = [session_options[key] for key in selected_session_keys]
responses = load_responses_from_sessions(selected_sessions)

# Create comparison data structure
comparison_data = []

for question in questions:
    qid = question["id"]
    q_data = {
        "id": qid,
        "question": question["question"],
        "answer": question.get("answer", ""),
        "topic": question.get("topic", []),
        "responses": []
    }
    
    # Add responses from selected sessions
    for response in responses:
        if qid in response["responses"]:
            q_data["responses"].append({
                "model_name": response["metadata"]["model_name"],
                "run_id": response["metadata"]["run_id"],
                "operator": response["metadata"]["operator"],
                "response": response["responses"][qid]
            })
    
    comparison_data.append(q_data)

# Display the data
st.header("Comparison Data")

# Add a download button for the JSON
json_str = json.dumps(comparison_data, indent=2)
st.download_button(
    label="Download Comparison Data (JSON)",
    data=json_str,
    file_name="comparison_data.json",
    mime="application/json"
)

# Display the data in an expandable format
for q_data in comparison_data:
    with st.expander(f"{q_data['id']}: {q_data['question'][:100]}..."):
        st.markdown(f"**Question:** {q_data['question']}")
        st.markdown(f"**Answer:** {q_data['answer']}")
        st.markdown(f"**Topics:** {', '.join(q_data['topic'])}")
        
        # Display responses
        st.subheader("Responses")
        for response in q_data["responses"]:
            st.markdown(f"""
            - **Model:** {response['model_name']}
            - **Run ID:** {response['run_id']}
            - **Operator:** {response['operator']}
            - **Response:** {response['response']}
            """) 