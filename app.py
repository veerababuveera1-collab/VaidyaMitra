import streamlit as st
import os
from crewai import Agent, Task, Crew
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

# --- UI Config & Styling ---
st.set_page_config(page_title="MediNode AI Pro", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stButton>button { background-color: #2c3e50; color: white; border-radius: 8px; height: 3em; width: 100%; }
    .report-box { background-color: white; padding: 25px; border-radius: 15px; border-left: 8px solid #3498db; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: #2c3e50; }
    .urgency-critical { background-color: #ffebee; border: 1px solid #ff1744; padding: 15px; border-radius: 10px; color: #b71c1c; font-weight: bold; }
    .urgency-normal { background-color: #e8f5e9; border: 1px solid #00c853; padding: 15px; border-radius: 10px; color: #1b5e20; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- State Definition ---
class AgentState(TypedDict):
    symptoms: str
    analysis: str
    urgency: str

# --- API Setup (Gemini) ---
def setup_llm():
    # Streamlit Secrets లేదా Sidebar నుండి కీని తీసుకుంటుంది
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("google_api_key")
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )

# --- Agentic Logic ---
def run_medical_crew(symptoms, llm):
    researcher = Agent(
        role='Medical Researcher',
        goal=f'Analyze symptoms: {symptoms}. Identify potential conditions.',
        backstory='Expert in clinical diagnostic patterns with access to medical knowledge.',
        llm=llm,
        allow_delegation=False,
        verbose=True
    )
    
    analyst = Agent(
        role='Triage Specialist',
        goal='Categorize the urgency of the symptoms.',
        backstory='Experienced emergency room nurse specializing in triage and patient prioritization.',
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

    task1 = Task(
        description=f"Analyze these symptoms: {symptoms}. Suggest top 3 likely causes.",
        agent=researcher,
        expected_output="A bulleted list of 3 potential medical conditions with brief reasons."
    )
    
    task2 = Task(
        description="Review findings and classify urgency as CRITICAL, MEDIUM, or LOW.",
        agent=analyst,
        expected_output="A final report with an Urgency Level and recommended next steps."
    )

    crew = Crew(agents=[researcher, analyst], tasks=[task1, task2])
    return crew.kickoff()

# --- LangGraph Node ---
def medical_node(state: AgentState):
    llm = setup_llm()
    result = run_medical_crew(state['symptoms'], llm)
    res_str = str(result)
    
    # Urgency Logic
    urgency = "Critical" if any(w in res_str.upper() for w in ["CRITICAL", "EMERGENCY", "IMMEDIATE"]) else "Normal"
    return {"analysis": res_str, "urgency": urgency}

# --- Graph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("analyze", medical_node)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", END)
app_graph = workflow.compile()

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.info("OpenAI Quota సమస్యను నివారించడానికి మేము Gemini AI ని వాడుతున్నాము.")
    
    if "GOOGLE_API_KEY" not in st.secrets:
        g_key = st.text_input("Enter Google API Key:", type="password")
        if g_key:
            st.session_state["google_api_key"] = g_key
    else:
        st.success("API Key loaded from Secrets! ✅")

    st.divider()
    st.write("Tech: CrewAI, LangGraph, Gemini 1.5 Flash")

# --- Main UI ---
st.title("🩺 VaidyaMitra AI: Agentic Medical Triage")
st.write("మీ ఆరోగ్య సమస్యలను వివరించండి, మా AI ఏజెంట్లు విశ్లేషిస్తాయి.")

user_input = st.text_area("Symptoms:", placeholder="ఉదా: మూడు రోజులుగా తలనొప్పి మరియు జ్వరం...", height=150)

if st.button("Start Analysis"):
    if not (st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("google_api_key")):
        st.error("దయచేసి Google API Key ని ఎంటర్ చేయండి!")
    elif not user_input:
        st.warning("ముందుగా లక్షణాలను టైప్ చేయండి.")
    else:
        with st.status("AI Agents are thinking...", expanded=True) as status:
            try:
                final_state = app_graph.invoke({"symptoms": user_input})
                status.update(label="Analysis Complete!", state="complete")
                
                st.subheader("📋 Final Report")
                st.markdown(f"<div class='report-box'>{final_state['analysis']}</div>", unsafe_allow_html=True)
                
                st.write("---")
                if final_state['urgency'] == "Critical":
                    st.markdown("<div class='urgency-critical'>🚨 URGENCY: CRITICAL - వెంటనే ఆసుపత్రికి వెళ్ళండి!</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='urgency-normal'>✅ URGENCY: NORMAL - సాధారణ జాగ్రత్తలు తీసుకోండి.</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
                status.update(label="Analysis Failed", state="error")

st.divider()
st.caption("Disclaimer: ఇది కేవలం AI ప్రోటోటైప్. అత్యవసర పరిస్థితుల్లో డాక్టరును సంప్రదించండి.")
