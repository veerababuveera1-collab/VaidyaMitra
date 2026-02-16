import streamlit as st
import os
from crewai import Agent, Task, Crew
from langgraph.graph import StateGraph, END
from typing import TypedDict

# --- UI Styling ---
st.set_page_config(page_title="MediNode AI Pro", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { border-radius: 10px; border: 1px solid #007bff; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 8px; font-weight: bold; }
    .report-box { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); border-left: 5px solid #007bff; }
    </style>
    """, unsafe_allow_html=True)

# --- State Definition ---
class AgentState(TypedDict):
    symptoms: str
    analysis: str
    urgency: str

# --- Agentic Logic (Fixed for Pydantic V2) ---
def run_medical_crew(symptoms):
    researcher = Agent(
        role='Medical Researcher',
        goal=f'Analyze symptoms: {symptoms}',
        backstory='Expert in medical science and diagnostic patterns.',
        allow_delegation=False,
        verbose=True
    )
    
    analyst = Agent(
        role='Triage Specialist',
        goal='Assess severity and provide medical guidance.',
        backstory='Specialized in emergency room triage and patient priority.',
        allow_delegation=False,
        verbose=True
    )

    # 'expected_output' ఇక్కడ తప్పనిసరి (Fix for ValidationError)
    task1 = Task(
        description=f"Analyze these symptoms: {symptoms}. Identify 3 potential causes.",
        agent=researcher,
        expected_output="A structured list of 3 potential medical conditions with brief explanations."
    )
    
    task2 = Task(
        description="Review researcher's findings and classify as CRITICAL, MEDIUM, or LOW urgency.",
        agent=analyst,
        expected_output="A final report with a clear Urgency Level and actionable advice for the patient."
    )

    crew = Crew(agents=[researcher, analyst], tasks=[task1, task2])
    return crew.kickoff()

# --- LangGraph Node ---
def medical_node(state: AgentState):
    result = run_medical_crew(state['symptoms'])
    res_str = str(result)
    # Urgency Check
    urg_check = res_str.upper()
    if any(word in urg_check for word in ["CRITICAL", "EMERGENCY", "IMMEDIATE"]):
        urgency = "Critical"
    else:
        urgency = "Normal"
    return {"analysis": res_str, "urgency": urgency}

# --- Workflow Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("analyze", medical_node)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", END)
app_graph = workflow.compile()

# --- Sidebar ---
with st.sidebar:
    st.header("🏥 System Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your sk-... key here")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.divider()
    st.info("ఈ సిస్టమ్ CrewAI ఏజెంట్లు మరియు LangGraph వర్క్‌ఫ్లోను ఉపయోగిస్తుంది.")

# --- Main Page ---
st.title("🩺 MediNode AI: Agentic Triage")
st.write("రోగి యొక్క లక్షణాలను కింద వివరించండి. మా AI ఏజెంట్లు వాటిని విశ్లేషిస్తాయి.")

user_input = st.text_area("Symptoms:", placeholder="ఉదా: నిన్నటి నుండి కడుపులో నొప్పి మరియు వాంతులు...", height=120)

if st.button("Start Analysis"):
    if not api_key:
        st.error("దయచేసి సైడ్‌బార్‌లో OpenAI API Key ని ఎంటర్ చేయండి!")
    elif not user_input:
        st.warning("లక్షణాలను నమోదు చేయండి.")
    else:
        with st.status("AI Agents are collaborating...", expanded=True) as status:
            st.write("🔍 Researcher is checking medical databases...")
            final_result = app_graph.invoke({"symptoms": user_input})
            status.update(label="Analysis Completed!", state="complete")
        
        # Displaying Results
        st.subheader("📋 Diagnostic Report")
        st.markdown(f"<div class='report-box'>{final_result['analysis']}</div>", unsafe_allow_html=True)
        
        st.write("---")
        if final_result['urgency'] == "Critical":
            st.error("🚨 **Urgency: CRITICAL** - వెంటనే డాక్టరును సంప్రదించండి!")
        else:
            st.success("✅ **Urgency: NORMAL** - సాధారణ జాగ్రత్తలు తీసుకోండి.")

st.caption("Disclaimer: ఇది కేవలం AI ప్రోటోటైప్ మాత్రమే. వైద్య సలహా కోసం డాక్టరును సంప్రదించండి.")
