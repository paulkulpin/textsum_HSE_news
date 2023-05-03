import streamlit as st
import requests
from requests.structures import CaseInsensitiveDict


def summarize(text, min_length, max_length):
    url = "http://localhost:8000/summarize/"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/x-www-form-urlencoded"

    data = f"text={text}, min_length={min_length}, max_length={max_length}"

    resp = requests.post(url, headers=headers, data=data)
    if resp.status_code != 200:
        raise ValueError("Failed to get summary from the API")
    return resp.json()["summary"]


st.title("Text Summarization with T5")
text = st.text_area("Enter text to be summarized:")

min_length = st.slider("Minimum summary length:", min_value=10, max_value=100, value=40, step=1)
max_length = st.slider("Maximum summary length:", min_value=50, max_value=200, value=150, step=1)

if st.button("Summarize"):
    summary = summarize(text, min_length=min_length, max_length=max_length)
    st.write(summary)
