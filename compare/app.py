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

# Function to load evaluation schema
@st.cache_data(ttl=60)
def load_evaluation_schema():
    try:
        schema_file_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="evaluation.json",
            repo_type="dataset",
            token=hf_token
        )
        with open(schema_file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load evaluation schema: {str(e)}")
        return None

# Function to validate evaluation data against schema
def validate_evaluation_data(data, schema):
    if not schema:
        return True, "No schema available for validation"
    
    def validate_against_schema(data, schema, path=""):
        if isinstance(schema, dict):
            # Check required fields from schema
            for field, field_schema in schema.items():
                if field not in data:
                    return False, f"Missing required field at {path}: {field}"
                # Recursively validate nested structure
                is_valid, message = validate_against_schema(data[field], field_schema, f"{path}.{field}")
                if not is_valid:
                    return False, message
        elif isinstance(schema, list) and len(schema) > 0:
            # For arrays, validate each item against the first schema item
            if not isinstance(data, list):
                return False, f"Expected array at {path}"
            for i, item in enumerate(data):
                is_valid, message = validate_against_schema(item, schema[0], f"{path}[{i}]")
                if not is_valid:
                    return False, message
        return True, "Validation successful"
    
    return validate_against_schema(data, schema)

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
comparison_data = {
    "session_ids": [session["session_id"] for session in selected_sessions],
    "items": []
}

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
                "session_id": response["session_id"],
                "response": response["responses"][qid]
            })
    
    comparison_data["items"].append(q_data)

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

# Add file upload and JSON paste functionality
st.header("Upload or Paste Evaluation Results")
uploaded_file = st.file_uploader("Upload JSON file with evaluation results", type=['json'])
json_text = st.text_area("Or paste JSON content here:", height=200)

# Load evaluation schema
evaluation_schema = load_evaluation_schema()

if uploaded_file is not None:
    try:
        uploaded_data = json.load(uploaded_file)
        # Validate against schema if available
        if evaluation_schema:
            is_valid, message = validate_evaluation_data(uploaded_data, evaluation_schema)
            if not is_valid:
                st.error(f"Validation error: {message}")
                st.stop()
        
        responses.append({
            "session_id": uploaded_data.get("session_id", "uploaded_file"),
            "metadata": {
                "model_name": uploaded_data.get("model_name", "Uploaded Model"),
                "run_id": uploaded_data.get("run_id", "uploaded"),
                "operator": uploaded_data.get("operator", "user")
            },
            "responses": uploaded_data.get("responses", {})
        })
        st.success("Successfully loaded uploaded JSON file!")
    except Exception as e:
        st.error(f"Error loading uploaded file: {str(e)}")

if json_text:
    try:
        pasted_data = json.loads(json_text)
        # Validate against schema if available
        if evaluation_schema:
            is_valid, message = validate_evaluation_data(pasted_data, evaluation_schema)
            if not is_valid:
                st.error(f"Validation error: {message}")
                st.stop()
        
        responses.append({
            "session_id": pasted_data.get("session_id", "pasted_json"),
            "metadata": {
                "model_name": pasted_data.get("model_name", "Pasted Model"),
                "run_id": pasted_data.get("run_id", "pasted"),
                "operator": pasted_data.get("operator", "user")
            },
            "responses": pasted_data.get("responses", {})
        })
        st.success("Successfully loaded pasted JSON content!")
    except Exception as e:
        st.error(f"Error loading pasted JSON: {str(e)}")

# Add upload to HF functionality
def upload_to_hf(comparison_data):
    try:
        # Create a unique filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"compare/comparison_{timestamp}.json"
        
        # Convert data to JSON string and then to bytes
        json_str = json.dumps(comparison_data, indent=2)
        json_bytes = json_str.encode('utf-8')
        
        # Upload to HF using bytes
        hf_api.upload_file(
            path_or_fileobj=json_bytes,
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        return True, filename
    except Exception as e:
        return False, str(e)

# Add upload button
if st.button("Upload Comparison to Hugging Face"):
    success, result = upload_to_hf(comparison_data)
    if success:
        st.success(f"Successfully uploaded comparison to {result}")
    else:
        st.error(f"Failed to upload comparison: {result}")

st.divider()
st.markdown("### 📋 Detailed Item Inspection (FYI Only)")
st.markdown("Below you can inspect individual questions and responses in detail. This view is for reference purposes only.")

# Display the data in an expandable format
for q_data in comparison_data["items"]:
    with st.expander(f"{q_data['id']}: {q_data['question'][:100]}..."):
        # Display metadata for all responses at the top
        if q_data["responses"]:
            st.subheader("Session Information")
            num_cols = min(len(q_data["responses"]), 3)  # Limit to 3 columns max
            metadata_cols = st.columns(num_cols)
            for idx, response in enumerate(q_data["responses"]):
                col_idx = idx % num_cols
                with metadata_cols[col_idx]:
                    st.markdown(f"""
                    **Session {idx + 1}**
                    - **Model:** {response['model_name']}
                    - **Run ID:** {response['run_id']}
                    - **Operator:** {response['operator']}
                    - **Session ID:** {response['session_id']}
                    """)
            
            st.divider()
        
        # Display question and answer
        st.subheader("Question and Answer")
        st.markdown(f"**Question:** {q_data['question']}")
        st.markdown(f"**Answer:** {q_data['answer']}")
        st.markdown(f"**Topics:** {', '.join(q_data['topic'])}")
        
        if q_data["responses"]:
            st.divider()
            
            # Display responses
            st.subheader("Responses")
            response_cols = st.columns(num_cols)
            for idx, response in enumerate(q_data["responses"]):
                col_idx = idx % num_cols
                with response_cols[col_idx]:
                    st.markdown(f"**Response {idx + 1}:**")
                    st.markdown(response['response']) 