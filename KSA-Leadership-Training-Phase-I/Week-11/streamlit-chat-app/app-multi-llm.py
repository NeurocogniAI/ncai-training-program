import streamlit as st
import google.generativeai as genai
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

st.title("🦜🔗 Multi-LLM Quickstart App")

# Sidebar LLM selection
llm_provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["Gemini", "Groq", "OpenAI"],
    index=0,
    help="Choose which LLM provider to use"
)

# Initialize API key variable
api_key = None

# Provider-specific API key input
if llm_provider == "Gemini":
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
elif llm_provider == "Groq":
    api_key = st.sidebar.text_input("Groq API Key", type="password")
elif llm_provider == "OpenAI":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

def generate_response(input_text):
    if llm_provider == "Gemini":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(input_text)
        st.info(response.text)
    elif llm_provider == "Groq":
        model = ChatGroq(
            temperature=0.7,
            api_key=api_key,
            model_name="deepseek-r1-distill-llama-70b"
            
        )
        st.info(model.invoke(input_text).content)
    elif llm_provider == "OpenAI":
        model = ChatOpenAI(
            temperature=0.7,
            api_key=api_key,
            model_name="gpt-3.5-turbo"
        )
        st.info(model.invoke(input_text))

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    
    if not api_key:
        st.warning(f"Please enter your {llm_provider} API key!", icon="⚠")
    if submitted and api_key:
        generate_response(text)