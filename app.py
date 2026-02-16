import streamlit as st
import os
from crewai import Agent, Task, Crew
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

# --- UI Config ---
st.set_page_config(page_title="VaidyaMitra AI", page_icon="🏥", layout="wide")

# --- State Definition ---
class AgentState(TypedDict):
    symptoms: str
    analysis: str
    urgency: str

# --- LLM Setup (Fix for 401 & 404 Errors) ---
def get_llm():
    # Google API Key సేకరణ
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("google_api_key")
    if not api_key:
        return None
    
    # OpenAI ఎర్రర్ రాకుండా ఉండటానికి ఖాళీ స్ట్రింగ్ సెట్ చేస్తున్నాం
    os.environ["OPENAI_API_KEY"] = "NA" 
    
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=api_key,
        temperature=0.2
    )

# --- Agentic Logic ---
def run_medical_crew(symptoms, llm):
    # ఏజెంట్లకు ఖచ్చితంగా Gemini LLM ని పాస్ చేయాలి
    researcher = Agent(
        role='Medical Researcher',
        goal=f'Identify conditions for: {symptoms}',
        backstory='Expert clinical diagnostic researcher.',
        llm=llm, # ఇక్కడ LLM ముఖ్యం
        allow_delegation=False,
        verbose=True
    )
    
    analyst = Agent(
        role='Triage Specialist',
        goal='Categorize medical urgency.',
        backstory='Senior ER triage specialist.',
        llm=llm, # ఇక్కడ LLM ముఖ్యం
        allow_delegation=False,
        verbose=True
    )

    task1 = Task(
        description=f"Analyze symptoms: {symptoms}. List 3 potential causes.",
        agent=researcher,
        expected_output="A list of 3 potential medical conditions."
    )
    
    task2 = Task(
        description="Determine if urgency is CRITICAL, MEDIUM, or LOW.",
        agent=analyst,
        expected_output="Final triage report with Urgency Level."
    )

    crew = Crew(agents=[researcher, analyst], tasks=[task1, task2])
    return crew.kickoff()

# --- LangGraph Node ---
def medical_node(state: AgentState):
    llm = get_llm()
    if not llm:
        return {"analysis": "API Key Missing", "urgency": "None"}
    
    result = run_medical_crew(state['symptoms'], llm)
    res_str = str(result)
    urgency = "Critical" if "CRITICAL" in res_str.upper() else "Normal"
    return {"analysis": res_str, "urgency": urgency}

# --- Graph Flow ---
workflow = StateGraph(AgentState)
workflow.add_node("analyze", medical_node)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", END)
app_graph = workflow.compile()

# --- Streamlit UI ---
st.title("🩺 VaidyaMitra: Agentic Medical Triage")

with st.sidebar:
    st.header("Settings")
    # Google API Key మాత్రమే అడుగుతున్నాం
    if "GOOGLE_API_KEY" not in st.secrets:
        g_key = st.text_input("Enter Google API Key (AIza...):", type="password")
        if g_key:
            st.session_state["google_api_key"] = g_key
    else:
        st.success("Google API Key loaded! ✅")

user_input = st.text_area("మీ లక్షణాలను వివరించండి:")

if st.button("Start AI Analysis"):
    if not (st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("google_api_key")):
        st.error("దయచేసి Google API Key ని ఎంటర్ చేయండి!")
    elif user_input:
        with st.status("AI Agents are working...", expanded=True):
            final_state = app_graph.invoke({"symptoms": user_input})
            st.subheader("Report")
            st.write(final_state['analysis'])
            
            if final_state['urgency'] == "Critical":
                st.error("🚨 URGENCY: CRITICAL")
            else:
                st.success("✅ URGENCY: NORMAL")
