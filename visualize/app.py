import streamlit as st
import json
import os
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import hashlib

st.set_page_config(layout="wide", page_title="Model Evaluation Comparison")

def check_auth():
    """Check if user is authenticated"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        try:
            users = st.secrets["users"]
        except:
            st.error("Authentication configuration not found in secrets!")
            return False
            
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username in users:
                stored_password = users[username]
                hashed_input = hashlib.sha256(password.encode()).hexdigest()
                
                if hashed_input == stored_password:
                    st.session_state.authenticated = True
                    st.experimental_rerun()
                else:
                    st.error("Invalid password")
            else:
                st.error("Invalid username")
        return False
    
    return True

def extract_schema(evaluation_data):
    """Extract the schema from evaluation data including all possible score types."""
    schema = {
        'score_types': set(),
        'metadata_fields': set(),
        'evaluation_fields': set()
    }
    
    # Extract from evaluation_metadata
    if 'evaluation_metadata' in evaluation_data:
        schema['metadata_fields'].update(evaluation_data['evaluation_metadata'].keys())
    
    # Extract from evaluation_criteria
    if 'evaluation_criteria' in evaluation_data:
        schema['score_types'].update(evaluation_data['evaluation_criteria'].keys())
    
    # Extract from evaluations
    if 'evaluations' in evaluation_data and evaluation_data['evaluations']:
        # Get all possible fields from the first evaluation
        first_eval = evaluation_data['evaluations'][0]
        schema['evaluation_fields'].update(first_eval.keys())
        
        # Get all possible score types from scores
        if 'scores' in first_eval:
            schema['score_types'].update(first_eval['scores'].keys())
    
    return schema

def load_evaluation_data(directory):
    """Load all evaluation JSON files from the compare directory."""
    evaluations = {}
    schemas = {}
    compare_dir = Path(directory)
    
    if not compare_dir.exists():
        st.error(f"Directory {directory} does not exist!")
        return {}, {}
    
    for file in compare_dir.glob("*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                model_name = file.stem
                evaluations[model_name] = data
                schemas[model_name] = extract_schema(data)
        except Exception as e:
            st.warning(f"Error loading {file}: {str(e)}")
    
    return evaluations, schemas

def create_comparison_table(evaluations, score_types):
    """Create a DataFrame comparing scores across models."""
    comparison_data = []
    
    # Get all unique question IDs
    all_questions = set()
    for eval_data in evaluations.values():
        all_questions.update(q['question_id'] for q in eval_data.get('evaluations', []))
    
    # Create comparison data
    for question_id in sorted(all_questions):
        row_data = {'question_id': question_id}
        
        # Get scores for each model
        for model_name, eval_data in evaluations.items():
            # Find the question in this model's evaluations
            question_data = next(
                (q for q in eval_data.get('evaluations', []) 
                 if q['question_id'] == question_id),
                None
            )
            
            if question_data:
                # Add scores for this model
                for score_type in score_types:
                    row_data[f"{model_name}_{score_type}"] = question_data['scores'].get(score_type, None)
            else:
                # Question not found in this model's evaluations
                for score_type in score_types:
                    row_data[f"{model_name}_{score_type}"] = None
        
        comparison_data.append(row_data)
    
    return pd.DataFrame(comparison_data)

def create_score_heatmap(df, score_type):
    """Create a heatmap of scores for a specific score type."""
    # Extract columns for the specific score type
    score_cols = [col for col in df.columns if col.endswith(f"_{score_type}")]
    model_names = [col.replace(f"_{score_type}", "") for col in score_cols]
    
    # Create heatmap data
    heatmap_data = df[score_cols].values
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=model_names,
        y=df['question_id'],
        colorscale='RdYlGn',
        zmin=1,
        zmax=10,
        colorbar=dict(title=f"{score_type.replace('_', ' ').title()} Score")
    ))
    
    fig.update_layout(
        title=f"{score_type.replace('_', ' ').title()} Comparison",
        xaxis_title="Model",
        yaxis_title="Question ID",
        height=800
    )
    
    return fig

def main():
    if not check_auth():
        return
        
    st.title("Model Evaluation Comparison")
    
    # Add logout button in sidebar
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.experimental_rerun()
    
    # Load evaluation data and schemas
    evaluations, schemas = load_evaluation_data("compare")
    
    if not evaluations:
        st.error("No evaluation data found!")
        return
    
    # Display schema information
    st.sidebar.header("Schema Information")
    for model_name, schema in schemas.items():
        with st.sidebar.expander(f"{model_name} Schema"):
            st.write("Score Types:", list(schema['score_types']))
            st.write("Metadata Fields:", list(schema['metadata_fields']))
            st.write("Evaluation Fields:", list(schema['evaluation_fields']))
    
    # Get common score types across all models
    common_score_types = set.intersection(*[schema['score_types'] for schema in schemas.values()])
    if not common_score_types:
        st.error("No common score types found across models!")
        return
    
    # Create comparison table
    comparison_df = create_comparison_table(evaluations, common_score_types)
    
    # Display summary statistics
    st.header("Summary Statistics")
    
    # Calculate average scores for each model
    summary_data = []
    for model_name in evaluations.keys():
        model_data = {'Model': model_name}
        for score_type in common_score_types:
            model_data[score_type.replace('_', ' ').title()] = comparison_df[f"{model_name}_{score_type}"].mean()
        summary_data.append(model_data)
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df.style.format("{:.2f}"))
    
    # Create and display heatmaps for each score type
    st.header("Score Heatmaps")
    for score_type in common_score_types:
        fig = create_score_heatmap(comparison_df, score_type)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed comparison table
    st.header("Detailed Comparison")
    st.dataframe(comparison_df)

if __name__ == "__main__":
    main()