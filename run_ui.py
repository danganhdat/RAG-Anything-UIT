import os
import sys

if __name__ == "__main__":
    from streamlit.web.cli import main as st_main
    sys.argv = ["streamlit", "run", __file__]
    st_main()
    sys.exit()

import requests
import streamlit as st

API_BASE = os.getenv("API_URL", "http://127.0.0.1:8000")
CHAT_URL = f"{API_BASE}/chat"
INGEST_URL = f"{API_BASE}/ingest/upload"
INFO_URL = f"{API_BASE}/system/info"

st.set_page_config(page_title="RAG-Anything Chat", page_icon="📚", layout="centered")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Cài đặt truy vấn")

    mode = st.selectbox(
        "Chế độ truy vấn",
        ["hybrid", "mix", "local", "global", "naive", "bypass"],
        index=0,
        help="hybrid = kết hợp, mix = đồ thị tri thức + vector, local = ngữ cảnh cục bộ, global = tri thức toàn cục",
    )
    top_k = st.slider("Số kết quả (Top K)", min_value=1, max_value=50, value=10)
    response_type = st.selectbox(
        "Định dạng trả lời",
        ["Multiple Paragraphs", "Single Paragraph", "Bullet Points"],
        index=0,
    )

    st.divider()
    st.header("Nhập tài liệu")
    uploaded = st.file_uploader(
        "Tải tài liệu lên",
        type=["pdf", "docx", "pptx", "xlsx", "png", "jpg", "jpeg", "txt", "md"],
    )
    if uploaded and st.button("Nhập tài liệu"):
        with st.spinner("Đang xử lý tài liệu..."):
            try:
                files = {"file": (uploaded.name, uploaded, uploaded.type or "application/octet-stream")}
                resp = requests.post(INGEST_URL, files=files, timeout=600)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Đã nhập thành công: {data['file_name']} ({data['elapsed_seconds']}s)")
                else:
                    st.error(f"Lỗi: {resp.status_code} — {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Không thể kết nối backend. Hãy chạy `python run_api.py` trước.")
            except Exception as exc:
                st.error(f"Lỗi không mong đợi: {exc}")

    st.divider()
    with st.expander("Thông tin hệ thống"):
        if st.button("Tải thông tin"):
            try:
                resp = requests.get(INFO_URL, timeout=10)
                if resp.status_code == 200:
                    st.json(resp.json())
                else:
                    st.error(f"Lỗi: {resp.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Không thể kết nối backend.")

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
st.title("📚 RAG-Anything Chat")
st.markdown("Đặt câu hỏi về nội dung tài liệu đã được nhập vào hệ thống!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Bạn muốn hỏi gì?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất ngữ cảnh và suy luận..."):
            try:
                payload = {
                    "query": prompt,
                    "mode": mode,
                    "top_k": top_k,
                    "response_type": response_type,
                    "conversation_history": st.session_state.messages[:-1],
                }
                resp = requests.post(CHAT_URL, json=payload, timeout=180)

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "Không có câu trả lời.")
                    st.markdown(answer)
                    st.caption(f"⏱ {data.get('elapsed_seconds', '?')}s · chế độ: {data.get('mode', mode)}")
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"Lỗi API: {resp.status_code} — {resp.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            except requests.exceptions.ConnectionError:
                error_msg = "Không thể kết nối backend. Hãy chạy `python run_api.py` trước."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except requests.exceptions.Timeout:
                error_msg = "Yêu cầu bị quá thời gian. Hệ thống RAG mất quá lâu để phản hồi."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as exc:
                error_msg = f"Lỗi không mong đợi: {exc}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
