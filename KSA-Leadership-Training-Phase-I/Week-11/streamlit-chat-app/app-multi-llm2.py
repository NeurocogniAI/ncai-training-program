import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

st.title("🦜🔗 Multi-LLM Quickstart App")

# Available LLM providers
LLM_PROVIDERS = {
    "Gemini": {
        "module": "gemini",
        "env_key": "GEMINI_API_KEY",
        "model": "gemini-2.0-flash"
    },
    "Groq": {
        "module": "groq",
        "env_key": "GROQ_API_KEY",
        "model": "deepseek-r1-distill-llama-70b"  # or "llama2-70b-4096"
    },
    "OpenAI": {
        "module": "openai",
        "env_key": "OPENAI_API_KEY",
        "model": "gpt-3.5-turbo"
    }
}

# Sidebar LLM selection
llm_provider = st.sidebar.selectbox(
    "Select LLM Provider",
    list(LLM_PROVIDERS.keys()),
    index=0
)

# Get API key from environment
api_key = os.getenv(LLM_PROVIDERS[llm_provider]["env_key"])

def generate_response(input_text):
    provider = LLM_PROVIDERS[llm_provider]
    
    if not api_key:
        st.error(f"Please set {provider['env_key']} in your .env file")
        return
    
    try:
        if llm_provider == "Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(provider["model"])
            response = model.generate_content(input_text)
            st.info(response.text)
            
        elif llm_provider == "Groq":
            model = ChatGroq(
                temperature=0.7,
                api_key=api_key,
                model_name=provider["model"]
            )
            st.info(model.invoke(input_text).content)
            
        elif llm_provider == "OpenAI":
            model = ChatOpenAI(
                temperature=0.7,
                api_key=api_key,
                model_name=provider["model"]
            )
            st.info(model.invoke(input_text).content)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        generate_response(text)