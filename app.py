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
    .stButton>button { background-color: #007bff; color: white; border-radius: 8px; font-weight: bold; width: 100%; }
    .report-box { background-color: #ffffff; padding: 25px; border-radius: 12px; box-shadow: 0px 4px 12px rgba(0,0,0,0.1); border-left: 6px solid #007bff; line-height: 1.6; color: #333; }
    .status-text { font-weight: 500; color: #555; }
    </style>
    """, unsafe_allow_html=True)

# --- State Definition ---
class AgentState(TypedDict):
    symptoms: str
    analysis: str
    urgency: str

# --- Agentic Logic (CrewAI) ---
def run_medical_crew(symptoms):
    # Researcher Agent
    researcher = Agent(
        role='Medical Researcher',
        goal=f'Analyze symptoms: {symptoms} and find potential causes.',
        backstory='Expert in clinical diagnostic patterns and medical literature.',
        allow_delegation=False,
        verbose=True
    )
    
    # Triage Specialist Agent
    analyst = Agent(
        role='Triage Specialist',
        goal='Assess severity and provide medical urgency classification.',
        backstory='Senior ER specialist focused on patient prioritization and safety.',
        allow_delegation=False,
        verbose=True
    )

    # Task 1: Research (Expected Output is mandatory for Pydantic V2)
    task1 = Task(
        description=f"Thoroughly analyze these symptoms: {symptoms}. List 3 most likely conditions.",
        agent=researcher,
        expected_output="A structured summary of 3 potential conditions with brief reasoning for each."
    )
    
    # Task 2: Triage
    task2 = Task(
        description="Review the research and classify the situation as CRITICAL, MEDIUM, or LOW urgency.",
        agent=analyst,
        expected_output="A final triage report including Urgency Level, brief explanation, and recommended next steps."
    )

    crew = Crew(agents=[researcher, analyst], tasks=[task1, task2])
    return crew.kickoff()

# --- LangGraph Node ---
def medical_node(state: AgentState):
    result = run_medical_crew(state['symptoms'])
    res_str = str(result)
    
    # Urgency detection logic
    urg_check = res_str.upper()
    if any(word in urg_check for word in ["CRITICAL", "EMERGENCY", "IMMEDIATE", "SEVERE"]):
        urgency = "Critical"
    else:
        urgency = "Normal"
        
    return {"analysis": res_str, "urgency": urgency}

# --- Workflow Graph Setup ---
workflow = StateGraph(AgentState)
workflow.add_node("analyze", medical_node)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", END)
app_graph = workflow.compile()

# --- Sidebar: API Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=80)
    st.header("⚙️ Configuration")
    
    # API Key Logic: Secrets vs Manual Input
    api_key_found = False
    
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        st.success("API Key loaded from Secrets! ✅")
        api_key_found = True
    else:
        user_key = st.text_input("Enter OpenAI API Key", type="password", help="sk-...")
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key
            api_key_found = True
        else:
            st.warning("⚠️ API Key missing! Please add to Secrets or enter here.")
    
    st.divider()
    st.markdown("""
    **Technical Stack:**
    - CrewAI (Multi-Agents)
    - LangGraph (Workflow)
    - OpenAI GPT Models
    """)

# --- Main Page UI ---
st.title("🩺 MediNode AI: Agentic Medical Triage")
st.write("మా AI ఏజెంట్ల బృందం మీ లక్షణాలను విశ్లేషించి, పరిస్థితి యొక్క తీవ్రతను తెలియజేస్తుంది.")

# Layout Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Patient Symptoms")
    user_input = st.text_area(
        "మీ ఆరోగ్య సమస్యలను ఇక్కడ వివరించండి:", 
        placeholder="ఉదా: విపరీతమైన ఛాతి నొప్పి మరియు శ్వాస తీసుకోవడంలో ఇబ్బంది...", 
        height=150
    )
    
    run_btn = st.button("Start AI Diagnostic Analysis")

with col2:
    st.subheader("Analysis Results")
    
    if run_btn:
        if not api_key_found:
            st.error("API Key దొరకలేదు. దయచేసి సెట్టింగ్స్ తనిఖీ చేయండి.")
        elif not user_input:
            st.warning("ముందుగా మీ లక్షణాలను టైప్ చేయండి.")
        else:
            with st.status("AI Agents are collaborating...", expanded=True) as status:
                st.write("🔍 **Medical Researcher** is scanning clinical patterns...")
                
                # Execute the Graph
                try:
                    final_result = app_graph.invoke({"symptoms": user_input})
                    status.update(label="Analysis Finished!", state="complete", expanded=False)
                    
                    # Report Display
                    st.markdown(f"<div class='report-box'>{final_result['analysis']}</div>", unsafe_allow_html=True)
                    
                    st.write("---")
                    if final_result['urgency'] == "Critical":
                        st.error("🚨 **URGENCY: CRITICAL** - వెంటనే వైద్య సహాయం తీసుకోండి (Consult a doctor immediately)!")
                    else:
                        st.success("✅ **URGENCY: NORMAL / STABLE** - సాధారణ వైద్య సలహాలు పాటించండి.")
                
                except Exception as e:
                    st.error(f"Error encountered: {e}")
                    status.update(label="Analysis Failed", state="error")

st.divider()
st.caption("⚠️ **Disclaimer:** This system is an AI research prototype and is NOT a substitute for professional medical diagnosis or emergency services.")
