import streamlit as st
import requests

# Constants
API_URL = "http://127.0.0.1:8000/chat"

# Page Configuration
st.set_page_config(
    page_title="RAG-Anything Chat",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("📚 RAG-Anything Chat")
st.markdown("Ask questions about the contents of your ingested PDF!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response placeholder with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and thinking..."):
            try:
                # Send the POST request to the FastAPI backend
                response = requests.post(API_URL, json={"query": prompt}, timeout=120)
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer provided.")
                    st.markdown(answer)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"API Error: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            except requests.exceptions.ConnectionError:
                error_msg = "Could not connect to the backend API. Please make sure `uvicorn api:app` is running."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except requests.exceptions.Timeout:
                error_msg = "The request timed out. The RAG system took too long to respond."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
