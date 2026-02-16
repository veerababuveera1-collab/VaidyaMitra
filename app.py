import streamlit as st
import os
from crewai import Agent, Task, Crew
from langgraph.graph import StateGraph, END
from typing import TypedDict

# --- UI Styling (Custom CSS) ---
st.set_page_config(page_title="AI Medical Agent Pro", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .report-card { background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; }
    </style>
    """, unsafe_allow_html=True)

# --- State Definition ---
class AgentState(TypedDict):
    symptoms: str
    analysis: str
    urgency: str

# --- Agentic Logic ---
def run_medical_crew(symptoms):
    researcher = Agent(
        role='Medical Researcher',
        goal=f'Identify potential conditions for: {symptoms}',
        backstory='Expert in clinical pathology and medical research databases.',
        verbose=False
    )
    
    analyst = Agent(
        role='Triage Specialist',
        goal='Determine medical urgency and provide next steps.',
        backstory='Senior ER triage nurse with 20 years of experience.',
        verbose=False
    )

    task1 = Task(description=f"Research these symptoms: {symptoms}. List top 3 possibilities.", agent=researcher)
    task2 = Task(description="Assess the research and categorize urgency as CRITICAL, MEDIUM, or LOW.", agent=analyst)

    crew = Crew(agents=[researcher, analyst], tasks=[task1, task2])
    return crew.kickoff()

# --- LangGraph Workflow ---
def medical_node(state: AgentState):
    result = run_medical_crew(state['symptoms'])
    res_str = str(result)
    urgency = "Critical" if any(word in res_str.upper() for word in ["CRITICAL", "EMERGENCY", "IMMEDIATE"]) else "Normal"
    return {"analysis": res_str, "urgency": urgency}

workflow = StateGraph(AgentState)
workflow.add_node("analyze", medical_node)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", END)
app_graph = workflow.compile()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=100)
    st.title("Control Panel")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your key to start the agents.")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    st.info("This system uses Multi-Agent Orchestration (CrewAI + LangGraph).")

# --- Main Interface ---
st.title("🩺 AI Agentic Medical Triage")
st.write("Enter patient symptoms below for an autonomous AI consultation.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Input")
    user_input = st.text_area("Describe symptoms in detail:", placeholder="e.g. Sharp pain in lower abdomen since 2 hours...", height=150)
    run_btn = st.button("Start AI Analysis")

with col2:
    st.subheader("Agent Live Status")
    if run_btn:
        if not api_key:
            st.error("Missing API Key!")
        elif not user_input:
            st.warning("Please describe symptoms.")
        else:
            with st.status("Agents are working...", expanded=True) as status:
                st.write("🔍 Researcher searching medical databases...")
                inputs = {"symptoms": user_input}
                final_state = app_graph.invoke(inputs)
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            # Display Results in a nice format
            st.markdown("### 📋 Final Medical Report")
            st.markdown(f"<div class='report-card'>{final_state['analysis']}</div>", unsafe_allow_html=True)
            
            if final_state['urgency'] == "Critical":
                st.error("🚨 **URGENCY: CRITICAL** - Seek immediate medical attention!")
            else:
                st.success("✅ **URGENCY: STABLE** - Consult a doctor at your convenience.")

st.divider()
st.caption("⚠️ Disclaimer: This is an AI prototype. Not for medical use.")
