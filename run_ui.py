import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

st.set_page_config(page_title="RAG-Anything Chat", page_icon="📚", layout="centered")

st.title("📚 RAG-Anything Chat")
st.markdown("Ask questions about the contents of your ingested PDF!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and thinking..."):
            try:
                resp = requests.post(API_URL, json={"query": prompt}, timeout=120)

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer provided.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"API Error: {resp.status_code} — {resp.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            except requests.exceptions.ConnectionError:
                error_msg = "Cannot connect to backend. Run `python run_api.py` first."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except requests.exceptions.Timeout:
                error_msg = "Request timed out. The RAG system took too long to respond."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as exc:
                error_msg = f"Unexpected error: {exc}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
