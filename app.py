import streamlit as st
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

# --- UI Config ---
st.set_page_config(page_title="VaidyaMitra AI", page_icon="🩺")

# --- State Definition ---
class AgentState(TypedDict):
    symptoms: str
    analysis: str

# --- LLM Setup ---
def get_llm():
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("google_api_key")
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )

# --- AI Logic (Direct Gemini Call) ---
def analyze_symptoms_node(state: AgentState):
    llm = get_llm()
    if not llm:
        return {"analysis": "API Key Missing!"}
    
    prompt = f"""
    You are a professional medical assistant. Analyze the following symptoms:
    Symptoms: {state['symptoms']}
    
    Provide:
    1. 3 Possible conditions.
    2. Urgency level (LOW, MEDIUM, CRITICAL).
    3. Suggested next steps.
    
    Disclaimer: Mention that this is AI-generated advice.
    """
    
    response = llm.invoke(prompt)
    return {"analysis": response.content}

# --- Graph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("medical_analysis", analyze_symptoms_node)
workflow.set_entry_point("medical_analysis")
workflow.add_edge("medical_analysis", END)
app_graph = workflow.compile()

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    if "GOOGLE_API_KEY" not in st.secrets:
        g_key = st.text_input("Enter Gemini API Key:", type="password")
        if g_key:
            st.session_state["google_api_key"] = g_key
    else:
        st.success("API Key Loaded! ✅")

# --- Main UI ---
st.title("🩺 VaidyaMitra: AI Medical Assistant")
st.write("మీ లక్షణాలను విశ్లేషించడానికి కింద ఉన్న బాక్స్‌లో టైప్ చేయండి.")

user_input = st.text_area("మీ ఆరోగ్య సమస్యలను వివరించండి:", placeholder="ఉదా: నాకు రెండు రోజులుగా తలనొప్పి మరియు జ్వరం ఉంది...")

if st.button("Analyze Now"):
    if not (st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("google_api_key")):
        st.error("దయచేసి API Key ని ఎంటర్ చేయండి!")
    elif user_input:
        with st.spinner("AI విశ్లేషిస్తోంది..."):
            try:
                final_output = app_graph.invoke({"symptoms": user_input})
                st.subheader("📋 విశ్లేషణ నివేదిక (Analysis Report)")
                st.write(final_output['analysis'])
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("దయచేసి మీ లక్షణాలను టైప్ చేయండి.")
