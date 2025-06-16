import streamlit as st
st.set_page_config(page_title="Boteval Prompt Manager", layout="wide")

import json
import datetime
import uuid
import io
from huggingface_hub import HfApi, hf_hub_download

# Load secrets
hf_token = st.secrets["hf"]["token"]
HF_REPO_ID = st.secrets["hf"]["repo_id"]

# Initialize Hugging Face API client
hf_api = HfApi(token=hf_token)

# Function to load prompts
@st.cache_data(ttl=60)
def load_prompts():
    try:
        prompts_file_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="system_prompts.json",
            repo_type="dataset",
            token=hf_token
        )
        with open(prompts_file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning("No prompts file found. Creating new one.")
        return {"prompts": {}}

# Function to save prompts
def save_prompts(prompts_data):
    prompts_json = json.dumps(prompts_data, indent=2)
    hf_api.upload_file(
        path_or_fileobj=io.BytesIO(prompts_json.encode()),
        path_in_repo="system_prompts.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )

# Load existing prompts
prompts_data = load_prompts()

st.title("System Prompt Manager")

# Import section
st.header("Import Prompts")
import_tab1, import_tab2 = st.tabs(["Upload JSON File", "Paste JSON"])

with import_tab1:
    uploaded_file = st.file_uploader("Upload a JSON file with prompts", type=['json'])
    if uploaded_file is not None:
        try:
            imported_data = json.load(uploaded_file)
            if st.button("Import from File"):
                # Merge imported prompts with existing ones
                for prompt_id, prompt in imported_data.get("prompts", {}).items():
                    if prompt_id not in prompts_data["prompts"]:
                        prompt["id"] = prompt_id
                        prompt["created_at"] = datetime.datetime.now().isoformat()
                        prompt["updated_at"] = datetime.datetime.now().isoformat()
                        prompts_data["prompts"][prompt_id] = prompt
                save_prompts(prompts_data)
                st.success("Prompts imported successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error loading JSON file: {str(e)}")

with import_tab2:
    json_text = st.text_area("Paste your JSON here", height=200)
    if json_text:
        try:
            imported_data = json.loads(json_text)
            if st.button("Import from Text"):
                # Merge imported prompts with existing ones
                for prompt_id, prompt in imported_data.get("prompts", {}).items():
                    if prompt_id not in prompts_data["prompts"]:
                        prompt["id"] = prompt_id
                        prompt["created_at"] = datetime.datetime.now().isoformat()
                        prompt["updated_at"] = datetime.datetime.now().isoformat()
                        prompts_data["prompts"][prompt_id] = prompt
                save_prompts(prompts_data)
                st.success("Prompts imported successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error parsing JSON: {str(e)}")

st.divider()

# Sidebar for prompt selection and creation
with st.sidebar:
    st.header("Prompt Management")
    
    # Add new prompt
    if st.button("Create New Prompt"):
        st.session_state.editing_prompt = {
            "id": str(uuid.uuid4()),
            "name": "",
            "description": "",
            "content": "",
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "tags": [],
            "metadata": {
                "author": "",
                "model": "",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            }
        }
    
    # List existing prompts
    st.subheader("Existing Prompts")
    for prompt_id, prompt in prompts_data["prompts"].items():
        if st.button(f"📝 {prompt['name']}", key=f"select_{prompt_id}"):
            st.session_state.editing_prompt = prompt

# Main content area
if "editing_prompt" in st.session_state:
    prompt = st.session_state.editing_prompt
    
    # Prompt editing form
    with st.form("prompt_form"):
        st.subheader("Edit Prompt")
        
        # Basic information
        prompt["name"] = st.text_input("Name", value=prompt["name"])
        prompt["description"] = st.text_area("Description", value=prompt["description"])
        prompt["content"] = st.text_area("Prompt Content", value=prompt["content"], height=300)
        
        # Tags
        tags_str = st.text_input("Tags (comma-separated)", value=", ".join(prompt["tags"]))
        prompt["tags"] = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
        
        # Metadata
        st.subheader("Metadata")
        col1, col2 = st.columns(2)
        with col1:
            prompt["metadata"]["author"] = st.text_input("Author", value=prompt["metadata"]["author"])
            prompt["metadata"]["model"] = st.text_input("Model", value=prompt["metadata"]["model"])
        with col2:
            prompt["metadata"]["parameters"]["temperature"] = st.number_input(
                "Temperature", 
                min_value=0.0, 
                max_value=2.0, 
                value=prompt["metadata"]["parameters"]["temperature"]
            )
            prompt["metadata"]["parameters"]["max_tokens"] = st.number_input(
                "Max Tokens", 
                min_value=1, 
                value=prompt["metadata"]["parameters"]["max_tokens"]
            )
        
        # Save button
        if st.form_submit_button("Save Prompt"):
            prompt["updated_at"] = datetime.datetime.now().isoformat()
            prompts_data["prompts"][prompt["id"]] = prompt
            save_prompts(prompts_data)
            st.success("Prompt saved successfully!")
            st.rerun()
        
        # Delete button
        if st.form_submit_button("Delete Prompt", type="primary"):
            if prompt["id"] in prompts_data["prompts"]:
                del prompts_data["prompts"][prompt["id"]]
                save_prompts(prompts_data)
                st.success("Prompt deleted successfully!")
                del st.session_state.editing_prompt
                st.rerun()

# Display all prompts in a table
if not st.session_state.get("editing_prompt"):
    st.subheader("All Prompts")
    
    # Create a table of prompts
    prompts_table = []
    for prompt_id, prompt in prompts_data["prompts"].items():
        prompts_table.append({
            "Name": prompt["name"],
            "Description": prompt["description"],
            "Tags": ", ".join(prompt["tags"]),
            "Model": prompt["metadata"]["model"],
            "Last Updated": prompt["updated_at"]
        })
    
    if prompts_table:
        st.dataframe(prompts_table, use_container_width=True)
    else:
        st.info("No prompts found. Create a new prompt using the sidebar.") 