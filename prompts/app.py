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
        st.warning("Could not load evaluation schema.")
        return None

# Function to save prompts
def save_prompts(prompts_data):
    prompts_json = json.dumps(prompts_data, indent=2)
    hf_api.upload_file(
        path_or_fileobj=io.BytesIO(prompts_json.encode()),
        path_in_repo="system_prompts.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )

# Load existing prompts and evaluation schema
prompts_data = load_prompts()
evaluation_schema = load_evaluation_schema()

st.title("System Prompt Manager")

# Create tabs for different functionalities
prompt_tab, generate_tab = st.tabs(["Prompt Management", "Generate System Prompt"])

with prompt_tab:
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

    # Main content area for prompt editing
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

with generate_tab:
    st.header("Generate System Prompt")
    
    # Display evaluation schema if available
    if evaluation_schema:
        st.subheader("Evaluation Schema")
        st.json(evaluation_schema)
    else:
        st.warning("No evaluation schema found. Please ensure evaluation.json exists in the repository.")
    
    # Custom prompt input
    st.subheader("Custom Prompt")
    custom_prompt = st.text_area("Enter your custom prompt instructions", height=200)
    
    # Model parameters
    st.subheader("Model Parameters")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.7)
    with col2:
        max_tokens = st.number_input("Max Tokens", min_value=1, value=1000)
    
    # Prompt name and description
    st.subheader("Prompt Details")
    prompt_name = st.text_input("Prompt Name", value="Evaluation System Prompt")
    prompt_description = st.text_area("Prompt Description", value="System prompt for evaluating responses using custom criteria and evaluation schema")
    
    # Generate button
    if st.button("Generate and Save System Prompt"):
        if not custom_prompt:
            st.error("Please enter a custom prompt.")
        elif not evaluation_schema:
            st.error("Evaluation schema is required.")
        else:
            # Combine custom prompt with evaluation schema
            system_prompt = f"""You are an AI assistant tasked with evaluating responses according to specific criteria.

Custom Instructions:
{custom_prompt}

Evaluation Schema:
{json.dumps(evaluation_schema, indent=2)}

Please evaluate the responses according to both the custom instructions and the evaluation schema provided above."""

            # Create prompt object
            prompt_id = str(uuid.uuid4())
            new_prompt = {
                "id": prompt_id,
                "name": prompt_name,
                "description": prompt_description,
                "content": system_prompt,
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "tags": ["evaluation", "system-prompt"],
                "metadata": {
                    "author": st.session_state.get("user_email", "unknown"),
                    "model": "GPT-4",
                    "parameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                }
            }
            
            # Add to prompts data
            prompts_data["prompts"][prompt_id] = new_prompt
            
            # Save to Hugging Face
            save_prompts(prompts_data)
            
            # Display success message and the generated prompt
            st.success("System prompt saved successfully!")
            st.subheader("Generated System Prompt")
            st.text_area("System Prompt", value=system_prompt, height=400)
            
            # Add copy button
            if st.button("Copy to Clipboard"):
                st.code(system_prompt)
                st.success("System prompt copied to clipboard!") 