import streamlit as st
import json
import os
import csv
import datetime

# Paths
QUESTIONS_FILE = os.path.join("..", "questions.json")
RESPONSES_FILE = os.path.join("data", "responses.csv")

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Load questions
with open(QUESTIONS_FILE, "r") as f:
    questions = json.load(f)

# Set page config
st.set_page_config(page_title="PhenixBB LLM Response Collector", layout="wide")

st.title("LLM Response Collector")

# Initialize session state for metadata
if "metadata" not in st.session_state:
    st.session_state.metadata = {
        "model_name": "GPT-4o",
        "run_id": "experiment_default",
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

# Add 'Clear Form' button
if st.button("Clear Form"):
    st.session_state.responses = {q['id']: "" for q in questions}
    st.session_state.metadata = {
        "model_name": "GPT-4o",
        "run_id": "experiment_default",
        "operator": os.getenv("USER", "unknown")
    }
    st.success("Form and metadata cleared!")

# Start form
with st.form("response_form"):
    st.header("Questions")

    for q in questions:
        st.session_state.responses[q['id']] = st.text_area(
            f"**{q['id']}**: {q['question']}", 
            height=200, 
            value=st.session_state.responses[q['id"]]
        )

    submitted = st.form_submit_button("Submit All Responses")

    if submitted:
        timestamp = datetime.datetime.now().isoformat()

        # Write each question separately (denormalized per question row)
        file_exists = os.path.isfile(RESPONSES_FILE)
        with open(RESPONSES_FILE, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "model_name", "run_id", "operator",
                                 "question_id", "question", "response"])

            for q in questions:
                row = [
                    timestamp,
                    st.session_state.metadata["model_name"],
                    st.session_state.metadata["run_id"],
                    st.session_state.metadata["operator"],
                    q['id'],
                    q['question'],
                    st.session_state.responses[q['id"]]
                ]
                writer.writerow(row)

        st.success("All responses saved successfully!")
        st.session_state.responses = {q['id']: "" for q in questions}
        st.session_state.metadata = {
            "model_name": "GPT-4o",
            "run_id": "experiment_default",
            "operator": os.getenv("USER", "unknown")
        }

