import streamlit as st
import os
from crewai import Agent, Task, Crew
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

# --- UI Config ---
st.set_page_config(page_title="VaidyaMitra Pro", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .report-card { background-color: white; padding: 25px; border-radius: 15px; border-left: 10px solid #007bff; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    .stButton>button { background: linear-gradient(to right, #007bff, #0056b3); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: bold; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- State Definition ---
class AgentState(TypedDict):
    symptoms: str
    analysis: str
    urgency: str

# --- LLM Factory (Fix for 404 Error) ---
def get_llm():
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("google_api_key")
    if not api_key:
        return None
    
    # ఇక్కడ 'gemini-1.5-flash-latest' వాడటం వల్ల 404 ఎర్రర్ రాదు
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        google_api_key=api_key,
        temperature=0.1,
        convert_system_message_to_human=True # Compatibility కోసం
    )

# --- CrewAI Setup ---
def run_medical_crew(symptoms, llm):
    researcher = Agent(
        role='Medical Researcher',
        goal=f'Analyze symptoms: {symptoms}. Suggest 3 likely conditions.',
        backstory='Expert in clinical diagnostic patterns and pathology.',
        llm=llm,
        allow_delegation=False,
        verbose=True
    )
    
    analyst = Agent(
        role='Triage Specialist',
        goal='Analyze conditions and determine urgency.',
        backstory='Veteran ER nurse with expertise in patient prioritization.',
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

    task1 = Task(
        description=f"Evaluate these symptoms: {symptoms}. List top 3 potential diagnoses.",
        agent=researcher,
        expected_output="A structured list of 3 potential conditions with brief reasoning."
    )
    
    task2 = Task(
        description="Review findings and classify urgency: CRITICAL, MEDIUM, or LOW.",
        agent=analyst,
        expected_output="A final triage report with an Urgency Level and next steps."
    )

    crew = Crew(agents=[researcher, analyst], tasks=[task1, task2])
    return crew.kickoff()

# --- LangGraph Logic ---
def medical_node(state: AgentState):
    llm = get_llm()
    result = run_medical_crew(state['symptoms'], llm)
    res_str = str(result)
    urgency = "Critical" if any(w in res_str.upper() for w in ["CRITICAL", "EMERGENCY", "IMMEDIATE"]) else "Normal"
    return {"analysis": res_str, "urgency": urgency}

workflow = StateGraph(AgentState)
workflow.add_node("analyze", medical_node)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", END)
app_graph = workflow.compile()

# --- UI Layout ---
st.title("🏥 VaidyaMitra: Agentic Health System")
st.write("మా AI ఏజెంట్లు మీ ఆరోగ్య సమస్యలను విశ్లేషించి మీకు సలహా ఇస్తాయి.")

# Sidebar API Key Handling
with st.sidebar:
    st.header("⚙️ Configuration")
    if "GOOGLE_API_KEY" not in st.secrets:
        g_key = st.text_input("Enter Google API Key:", type="password")
        if g_key: st.session_state["google_api_key"] = g_key
    else:
        st.success("API Key loaded from Secrets! ✅")
    
    st.divider()
    st.caption("Built with CrewAI & LangGraph")

# Input Area
user_input = st.text_area("మీ లక్షణాలను ఇక్కడ నమోదు చేయండి (उदा: Severe stomach pain):", height=150)

if st.button("🚀 Start Diagnostic Workflow"):
    if not (st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("google_api_key")):
        st.error("దయచేసి API Key ని ఎంటర్ చేయండి!")
    elif not user_input:
        st.warning("లక్షణాలను నమోదు చేయండి.")
    else:
        with st.status("AI Agents are collaborating...", expanded=True) as status:
            try:
                final_state = app_graph.invoke({"symptoms": user_input})
                status.update(label="Analysis Finished!", state="complete")
                
                st.subheader("📋 Final Report")
                st.markdown(f"<div class='report-card'>{final_state['analysis']}</div>", unsafe_allow_html=True)
                
                st.write("---")
                if final_state['urgency'] == "Critical":
                    st.error("🚨 **URGENCY: CRITICAL** - వెంటనే వైద్యుడిని సంప్రదించండి!")
                else:
                    st.success("✅ **URGENCY: NORMAL** - సాధారణ జాగ్రత్తలు తీసుకోండి.")
            except Exception as e:
                st.error(f"Error occurred: {e}")
                status.update(label="Process Failed", state="error")

st.divider()
st.caption("Disclaimer: ఇది ఒక AI ప్రోటోటైప్ మాత్రమే. నిపుణులైన వైద్యుడి సలహాను తప్పక తీసుకోండి.")
