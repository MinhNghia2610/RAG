import streamlit as st
from src.retriever import retrieve
from src.generator import generate_answer

st.title("💬 Chatbot Thông Minh")

query = st.text_input("Nhập câu hỏi của bạn:")

if st.button("Gửi"):
    contexts = retrieve(query)
    answer = generate_answer(contexts, query)
    st.write("**Trả lời:**", answer)
