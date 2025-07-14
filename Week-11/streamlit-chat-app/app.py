import streamlit as st
import google.generativeai as genai

st.title("🦜🔗 Quickstart App (Gemini)")

gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

def generate_response(input_text):
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(input_text)
    st.info(response.text)
    

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if not gemini_api_key:
        st.warning("Please enter your Gemini API key!", icon="⚠")
    if submitted and gemini_api_key:
        generate_response(text)