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

# Function to load all annotations
@st.cache_data(ttl=60)
def load_annotations():
    annotations = []
    # List all files in the annotate directory
    files = hf_api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
    annotation_files = [f for f in files if f.startswith("annotate/annotation-")]
    
    for file in annotation_files:
        try:
            file_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=file,
                repo_type="dataset",
                token=hf_token
            )
            with open(file_path, "r") as f:
                annotations.append(json.load(f))
        except Exception as e:
            st.warning(f"Could not load annotation file {file}: {str(e)}")
    
    return annotations

# Function to load all responses
@st.cache_data(ttl=60)
def load_responses():
    responses = []
    # List all files in the gather directory
    files = hf_api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
    response_files = [f for f in files if f.startswith("gather/submission-")]
    
    for file in response_files:
        try:
            file_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=file,
                repo_type="dataset",
                token=hf_token
            )
            with open(file_path, "r") as f:
                responses.append(json.load(f))
        except Exception as e:
            st.warning(f"Could not load response file {file}: {str(e)}")
    
    return responses

# Load all data
questions = load_questions()
annotations = load_annotations()
responses = load_responses()

# Create comparison data structure
comparison_data = []

for question in questions:
    qid = question["id"]
    q_data = {
        "id": qid,
        "question": question["question"],
        "answer": question.get("answer", ""),
        "topic": question.get("topic", []),
        "annotations": [],
        "responses": []
    }
    
    # Add annotations
    for annotation in annotations:
        if qid in annotation.get("annotations", {}):
            q_data["annotations"].append({
                "annotator": annotation["annotator"],
                "benchmark": annotation["annotations"][qid].get("benchmark", ""),
                "quality": annotation["annotations"][qid].get("quality", 0)
            })
    
    # Add responses
    for response in responses:
        if qid in response.get("responses", {}):
            q_data["responses"].append({
                "model_name": response["model_name"],
                "run_id": response["run_id"],
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
        
        # Display annotations
        st.subheader("Annotations")
        for annotation in q_data["annotations"]:
            benchmark = annotation['benchmark'] if annotation['benchmark'] else "Not provided"
            st.markdown(f"""
            - **Annotator:** {annotation['annotator']}
            - **Benchmark:** {benchmark}
            - **Quality:** {annotation['quality']}
            """)
        
        # Display responses
        st.subheader("Responses")
        for response in q_data["responses"]:
            st.markdown(f"""
            - **Model:** {response['model_name']}
            - **Run ID:** {response['run_id']}
            - **Response:** {response['response']}
            """) 