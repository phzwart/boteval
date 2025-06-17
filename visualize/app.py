import streamlit as st
import json
import os
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import hashlib

st.set_page_config(layout="wide", page_title="Model Evaluation Comparison")

def get_hf_token():
    """Get HF token from Streamlit secrets"""
    try:
        return st.secrets["hf"]["token"]
    except:
        st.error("Hugging Face token not found in Streamlit secrets! Please add token under [hf] section.")
        return None

def get_repo_id():
    """Get repo_id from Streamlit secrets or user input"""
    try:
        return st.secrets["hf"]["repo_id"]
    except:
        return None

def check_auth():
    """Check if user is authenticated"""
    # Skip authentication if running locally or if secrets aren't configured
    if os.environ.get('STREAMLIT_SERVER_RUNNING_LOCALLY', 'false').lower() == 'true':
        return True
        
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        try:
            users = st.secrets["authorized_users"]
        except:
            # If secrets aren't configured, skip authentication
            st.warning("Authentication not configured - proceeding without login")
            return True
            
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username in users:
                stored_password = users[username]
                if password == stored_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid password")
            else:
                st.error("Invalid username")
        return False
    
    return True

@st.cache_data(ttl=3600)  # Cache for 1 hour
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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_evaluation_data(repo_id, directory="compare", selected_files=None):
    """Load selected evaluation JSON files from the Hugging Face repo."""
    evaluations = {}
    schemas = {}
    
    # Get HF token
    token = get_hf_token()
    if not token:
        st.error("Hugging Face token not found in Streamlit secrets! Please add token under [hf] section.")
        return {}, {}
    
    # Initialize HF API
    api = HfApi(token=token)
    
    try:
        # List files in the directory
        files = api.list_repo_files(repo_id, repo_type="dataset")
        json_files = [f for f in files if f.startswith(f"{directory}/") and f.endswith(".json")]
        
        # Filter files if specific ones are selected
        if selected_files:
            json_files = [f for f in json_files if Path(f).stem in selected_files]
        
        for file_path in json_files:
            try:
                # Download file
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=file_path,
                    token=token
                )
                
                # Load data
                with open(local_path, 'r') as f:
                    data = json.load(f)
                    model_name = Path(file_path).stem
                    evaluations[model_name] = data
                    schemas[model_name] = extract_schema(data)
            except Exception as e:
                st.warning(f"Error loading {file_path}: {str(e)}")
    
    except Exception as e:
        st.error(f"Error accessing Hugging Face repository: {str(e)}")
        return {}, {}
    
    return evaluations, schemas

@st.cache_data(ttl=3600)  # Cache for 1 hour
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
            # Get evaluator from metadata
            evaluator = eval_data.get('evaluation_metadata', {}).get('evaluator', 'unknown')
            row_data[f"{model_name}_evaluator"] = evaluator
            
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

@st.cache_data(ttl=3600)  # Cache for 1 hour
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
        colorbar=dict(title=f"{score_type.replace('_', ' ').title()} Score"),
        xgap=1,  # Add gap between x-axis elements
        ygap=1,  # Add gap between y-axis elements
        text=heatmap_data,  # Add text values
        texttemplate="%{text:.1f}",  # Format text to 1 decimal place
        textfont={"color": "black"},  # Set text color to black
    ))
    
    # Add black grid lines
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='black',
        tickfont=dict(color='black'),  # Set axis text color to black
        title_font=dict(color='black')  # Set axis title color to black
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='black',
        tickfont=dict(color='black'),  # Set axis text color to black
        title_font=dict(color='black')  # Set axis title color to black
    )
    
    fig.update_layout(
        title=dict(
            text=f"{score_type.replace('_', ' ').title()} Comparison",
            font=dict(color='black')  # Set title color to black
        ),
        xaxis_title="Model",
        yaxis_title="Question ID",
        height=800,
        plot_bgcolor='white',  # Set background to white
        paper_bgcolor='white',  # Set paper background to white
        font=dict(color='black')  # Set default font color to black
    )
    
    return fig

@st.cache_data(ttl=3600)  # Cache for 1 hour
def create_score_histogram(_model_names, df, score_type):
    """Create a histogram of scores for a specific score type across models."""
    fig = go.Figure()
    
    for model_name in _model_names:
        col_name = f"{model_name}_{score_type}"
        scores = df[col_name].dropna()
        
        fig.add_trace(go.Histogram(
            x=scores,
            name=model_name,
            opacity=0.7,
            nbinsx=20,
            histnorm='probability'
        ))
    
    fig.update_layout(
        title=dict(
            text=f"{score_type.replace('_', ' ').title()} Score Distribution",
            font=dict(color='black', size=16)
        ),
        xaxis=dict(
            title=dict(
                text="Score",
                font=dict(color='black', size=14)
            ),
            tickfont=dict(color='black', size=12)
        ),
        yaxis=dict(
            title=dict(
                text="Probability",
                font=dict(color='black', size=14)
            ),
            tickfont=dict(color='black', size=12)
        ),
        barmode='overlay',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(color="black", size=12)
        )
    )
    
    return fig

def main():
    if not check_auth():
        return
        
    st.title("Model Evaluation Comparison")
    
    # Add logout button in sidebar
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    # Get repository ID from secrets or user input
    default_repo = get_repo_id() or "phzwart/boteval-phenixbb"
    repo_id = st.sidebar.text_input("Hugging Face Repository ID", default_repo)
    
    # Initialize HF API to get available files
    token = get_hf_token()
    if token:
        api = HfApi(token=token)
        try:
            files = api.list_repo_files(repo_id, repo_type="dataset")
            json_files = [Path(f).stem for f in files if f.startswith("compare/") and f.endswith(".json")]
            
            # Add file selector in sidebar
            st.sidebar.header("Select Evaluations")
            selected_files = st.sidebar.multiselect(
                "Choose evaluation files to compare",
                options=json_files,
                default=json_files  # Default to all files
            )
            
            # Load evaluation data and schemas
            evaluations, schemas = load_evaluation_data(repo_id, selected_files=selected_files)
            
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
            
            # Add question exclusion in sidebar
            st.sidebar.header("Question Filtering")
            all_questions = comparison_df['question_id'].tolist()
            excluded_questions = st.sidebar.multiselect(
                "Exclude questions from summary statistics",
                options=all_questions,
                default=[]
            )
            
            # Filter comparison dataframe for summary statistics and visualizations
            filtered_df = comparison_df[~comparison_df['question_id'].isin(excluded_questions)]
            
            # Display summary statistics
            st.header("Summary Statistics")
            if excluded_questions:
                st.info(f"Excluded {len(excluded_questions)} questions from summary statistics")
            
            # Calculate statistics for each model
            summary_data = []
            for model_name in evaluations.keys():
                # Get evaluator from the first row (it's the same for all rows)
                evaluator = comparison_df[f"{model_name}_evaluator"].iloc[0]
                model_data = {
                    'Model': model_name,
                    'Evaluator': evaluator
                }
                for score_type in common_score_types:
                    col_name = f"{model_name}_{score_type}"
                    scores = filtered_df[col_name]  # Use filtered data
                    # Format the statistics as a string
                    stats_str = f"Q25: {scores.quantile(0.25):.2f} | Median: {scores.median():.2f} | Q75: {scores.quantile(0.75):.2f}"
                    model_data[score_type.replace('_', ' ').title()] = stats_str
                summary_data.append(model_data)
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
            
            # Add download buttons
            col1, col2 = st.columns(2)
            with col1:
                # Download summary statistics
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary Statistics",
                    data=summary_csv,
                    file_name="summary_statistics.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download full comparison table
                full_csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="Download Full Comparison Table",
                    data=full_csv,
                    file_name="full_comparison.csv",
                    mime="text/csv"
                )
            
            # Create and display heatmaps for each score type
            st.header("Score Heatmaps")
            for score_type in common_score_types:
                # Use full comparison_df for heatmap to show all questions
                fig = create_score_heatmap(comparison_df, score_type)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add histogram for this score type using filtered data
                st.subheader(f"{score_type.replace('_', ' ').title()} Score Distribution")
                hist_fig = create_score_histogram(tuple(evaluations.keys()), filtered_df, score_type)
                st.plotly_chart(hist_fig, use_container_width=True)
            
            # Display detailed comparison table
            st.header("Detailed Comparison")
            st.dataframe(comparison_df)
        except Exception as e:
            st.error(f"Error accessing Hugging Face repository: {str(e)}")

if __name__ == "__main__":
    main()