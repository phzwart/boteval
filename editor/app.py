import streamlit as st
import json
import io
from huggingface_hub import HfApi, hf_hub_download
import uuid

# Load Hugging Face token and repo ID from Streamlit Secrets
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

# Load questions.json from Hugging Face Hub
@st.cache_data(ttl=60, show_spinner=False)
def load_questions():
    questions_file_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="questions.json",
        repo_type="dataset",
        token=hf_token
    )
    with open(questions_file_path, "r") as f:
        questions = json.load(f)
    return questions

questions = load_questions()

# Session state to track edits
if "questions" not in st.session_state:
    st.session_state.questions = questions

st.title("Boteval Questions Editor")

# --- Add new question ---
st.subheader("Add New Question")

with st.form("add_question_form"):
    new_id = st.text_input("ID (unique)", value=f"Q{str(uuid.uuid4())[:8]}")
    new_question = st.text_area("Question Text", height=100)
    new_topics = st.text_input("Topics (comma separated)")

    submitted = st.form_submit_button("Add Question")

    if submitted:
        if not any(q["id"] == new_id for q in st.session_state.questions):
            st.session_state.questions.append({
                "id": new_id,
                "question": new_question,
                "topic": [t.strip() for t in new_topics.split(",") if t.strip()]
            })
            st.success("Question added.")
        else:
            st.error("ID already exists.")

st.divider()

# --- Display existing questions ---
st.subheader("Edit Existing Questions")

for idx, q in enumerate(st.session_state.questions):
    with st.expander(f"{q['id']}: {q['question'][:60]}"):
        edited_question = st.text_area("Edit Question", value=q["question"], key=f"question_{idx}")
        edited_topics = st.text_input(
            "Edit Topics (comma separated)",
            value=", ".join(q.get("topic", [])),
            key=f"topics_{idx}"
        )

        if st.button("Update", key=f"update_{idx}"):
            st.session_state.questions[idx]["question"] = edited_question
            st.session_state.questions[idx]["topic"] = [
                t.strip() for t in edited_topics.split(",") if t.strip()
            ]
            st.success("Question updated.")

        if st.button("Delete", key=f"delete_{idx}"):
            st.session_state.questions.pop(idx)
            st.success("Question deleted.")
            st.rerun()  # Refresh after delete

st.divider()

# --- Save all changes ---
if st.button("Save All Changes to Hugging Face"):
    updated_json = json.dumps(st.session_state.questions, indent=2)
    hf_api.upload_file(
        path_or_fileobj=io.BytesIO(updated_json.encode()),
        path_in_repo="questions.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )
    st.success("Updated questions.json saved to Hugging Face Hub!")

