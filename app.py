import streamlit as st
import os

st.set_page_config(
    page_title="Boteval Master App",
    page_icon="🤖",
    layout="wide"
)

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Authentication function
def authenticate_user(email, password):
    authorized_users = st.secrets.get("authorized_users", {})
    return email in authorized_users and authorized_users[email] == password

# Login form
if not st.session_state.authenticated:
    st.title("Boteval Login")
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

# Main app content
st.title("🤖 Boteval Master App")
st.markdown("Welcome to the Boteval platform! Select a component below to get started.")

# Create columns for the app cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Data Collection & Management")
    
    # Gather App
    st.markdown("#### 📝 Response Collector")
    st.markdown("Collect and manage LLM responses for evaluation.")
    if st.button("Launch Response Collector", key="gather"):
        os.system("streamlit run gather/app.py")
    
    # Prompts App
    st.markdown("#### 💭 Prompt Manager")
    st.markdown("Create and manage system prompts for LLMs.")
    if st.button("Launch Prompt Manager", key="prompts"):
        os.system("streamlit run prompts/app.py")
    
    # Editor App
    st.markdown("#### ✏️ Response Editor")
    st.markdown("Edit and refine LLM responses.")
    if st.button("Launch Response Editor", key="editor"):
        os.system("streamlit run editor/app.py")

with col2:
    st.markdown("### Evaluation & Analysis")
    
    # Compare App
    st.markdown("#### 🔄 Response Comparison")
    st.markdown("Compare different LLM responses side by side.")
    if st.button("Launch Response Comparison", key="compare"):
        os.system("streamlit run compare/app.py")
    
    # Annotate App
    st.markdown("#### 📋 Annotation Tool")
    st.markdown("Annotate and evaluate LLM responses.")
    if st.button("Launch Annotation Tool", key="annotate"):
        os.system("streamlit run annotate/app.py")
    
    # Visualize App
    st.markdown("#### 📊 Evaluation Visualization")
    st.markdown("Visualize and analyze evaluation results.")
    if st.button("Launch Visualization", key="visualize"):
        os.system("streamlit run visualize/app.py")

# Add logout button in sidebar
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

# Add user info in sidebar
st.sidebar.markdown(f"Logged in as: {st.session_state.user_email}") 