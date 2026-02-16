import streamlit as st
import os
from crewai import Agent, Task, Crew
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

# --- UI Config ---
st.set_page_config(page_title="VaidyaMitra AI", page_icon="🩺", layout="wide")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .report-card { background-color: white; padding: 25px; border-radius: 15px; border-left: 10px solid #007bff; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    .urgency-badge { padding: 10px 20px; border-radius: 50px; font-weight: bold; text-align: center; display: inline-block; }
    .critical { background-color: #ff4b4b; color: white; }
    .normal { background-color: #28a745; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- State Definition ---
class AgentState(TypedDict):
    symptoms: str
    analysis: str
    urgency: str

# --- LLM Setup (Gemini Fix) ---
def get_llm():
    # Secrets లేదా Session State నుండి కీని తీసుకుంటుంది
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("google_api_key")
    if not api_key:
        return None
    
    # మోడల్ పేరును 'gemini-1.5-flash' గా స్థిరపరిచాను (Fix for 404 error)
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )

# --- CrewAI Logic ---
def run_medical_crew(symptoms, llm):
    researcher = Agent(
        role='Medical Researcher',
        goal=f'Analyze symptoms: {symptoms}. Provide 3 possible conditions.',
        backstory='Expert clinical researcher with a deep understanding of medical diagnosis.',
        llm=llm,
        allow_delegation=False,
        verbose=True
    )
    
    analyst = Agent(
        role='Triage Specialist',
        goal='Analyze the researcher’s findings and determine urgency.',
        backstory='Veteran ER specialist skilled at triage and determining medical priority.',
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

    task1 = Task(
        description=f"Review the symptoms: {symptoms}. List 3 potential causes with brief details.",
        agent=researcher,
        expected_output="A list of 3 potential medical conditions based on the symptoms provided."
    )
    
    task2 = Task(
        description="Categorize the urgency as CRITICAL, MEDIUM, or LOW based on the findings.",
        agent=analyst,
        expected_output="A final report starting with 'URGENCY LEVEL: [LEVEL]' and followed by advice."
    )

    crew = Crew(agents=[researcher, analyst], tasks=[task1, task2])
    return crew.kickoff()

# --- LangGraph Node ---
def medical_node(state: AgentState):
    llm = get_llm()
    result = run_medical_crew(state['symptoms'], llm)
    res_str = str(result)
    
    # Simple logic to detect urgency
    urgency = "Critical" if any(w in res_str.upper() for w in ["CRITICAL", "EMERGENCY", "IMMEDIATE"]) else "Normal"
    return {"analysis": res_str, "urgency": urgency}

# --- Graph Flow ---
workflow = StateGraph(AgentState)
workflow.add_node("analyze", medical_node)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", END)
app_graph = workflow.compile()

# --- Sidebar ---
with st.sidebar:
    st.title("🏥 Control Center")
    if "GOOGLE_API_KEY" not in st.secrets:
        g_key = st.text_input("Enter Gemini API Key:", type="password")
        if g_key:
            st.session_state["google_api_key"] = g_key
            st.success("Key set successfully!")
    else:
        st.success("API Key loaded from Secrets! ✅")
    
    st.divider()
    st.markdown("### How it works?")
    st.write("1. 🔍 **Researcher**: Scans symptoms.")
    st.write("2. ⚖️ **Analyst**: Determines urgency.")
    st.write("3. 📊 **Graph**: Manages the flow.")

# --- Main UI ---
st.title("🩺 VaidyaMitra: Agentic Medical System")
st.write("మా AI ఏజెంట్ల బృందం మీ లక్షణాలను విశ్లేషించడానికి సిద్ధంగా ఉంది.")

user_input = st.text_area("మీ ఆరోగ్య సమస్యలను వివరించండి (English or Telugu):", height=150, placeholder="Ex: Severe headache and blurred vision for 2 days...")

if st.button("🚀 Start Diagnostic Process"):
    if not (st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("google_api_key")):
        st.error("దయచేసి సైడ్‌బార్‌లో Google API Key ఎంటర్ చేయండి!")
    elif not user_input:
        st.warning("ముందుగా మీ లక్షణాలను నమోదు చేయండి.")
    else:
        with st.status("AI Agents are analyzing your health...", expanded=True) as status:
            try:
                # Run the Agentic Workflow
                final_state = app_graph.invoke({"symptoms": user_input})
                status.update(label="Analysis Finished!", state="complete")
                
                # Result Display
                st.subheader("📋 Diagnostic Summary")
                st.markdown(f"<div class='report-card'>{final_state['analysis']}</div>", unsafe_allow_html=True)
                
                st.write("---")
                if final_state['urgency'] == "Critical":
                    st.markdown("<div class='urgency-badge critical'>🚨 URGENCY: CRITICAL - Consult a doctor immediately!</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='urgency-badge normal'>✅ URGENCY: NORMAL - Follow standard health care.</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error occurred: {e}")
                status.update(label="Analysis Failed", state="error")

st.divider()
st.caption("⚠️ Disclaimer: This is an AI research prototype. Not a substitute for professional medical diagnosis.")
