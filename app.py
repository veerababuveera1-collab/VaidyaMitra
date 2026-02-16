import streamlit as st
import os
from crewai import Agent, Task, Crew
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

# --- UI Setup ---
st.set_page_config(page_title="VaidyaMitra AI", page_icon="🩺")

# --- Agent State ---
class AgentState(TypedDict):
    symptoms: str
    analysis: str
    urgency: str

# --- API & LLM Setup ---
def get_llm():
    # Secrets నుండి లేదా Sidebar నుండి కీని పొందండి
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("google_api_key")
    
    if not api_key:
        return None
    
    # OpenAI ఎర్రర్స్ రాకుండా నివారించడం
    os.environ["OPENAI_API_KEY"] = "NA"

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # లేదా "gemini-1.5-flash-latest"
        google_api_key=api_key,
        temperature=0.1
    )

# --- CrewAI Logic ---
def run_medical_crew(symptoms, llm):
    # ఏజెంట్లకు llm ఆబ్జెక్ట్‌ను స్పష్టంగా పాస్ చేయాలి
    researcher = Agent(
        role='Medical Researcher',
        goal='Analyze symptoms and identify potential conditions.',
        backstory='Expert clinical researcher with deep medical knowledge.',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    analyst = Agent(
        role='Triage Specialist',
        goal='Determine the urgency of the medical situation.',
        backstory='Specialist in ER triage and patient prioritization.',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    t1 = Task(
        description=f"Analyze: {symptoms}. List top 3 causes.",
        agent=researcher,
        expected_output="A list of 3 potential causes."
    )
    
    t2 = Task(
        description="Determine if urgency is CRITICAL or NORMAL.",
        agent=analyst,
        expected_output="Final triage report with URGENCY level."
    )

    crew = Crew(agents=[researcher, analyst], tasks=[t1, t2])
    return crew.kickoff()

# --- LangGraph Node ---
def medical_node(state: AgentState):
    llm = get_llm()
    if not llm:
        return {"analysis": "Missing API Key", "urgency": "None"}
    
    result = run_medical_crew(state['symptoms'], llm)
    res_str = str(result)
    
    urgency = "Critical" if "CRITICAL" in res_str.upper() else "Normal"
    return {"analysis": res_str, "urgency": urgency}

# --- Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("analyze", medical_node)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", END)
app_graph = workflow.compile()

# --- Sidebar & UI ---
with st.sidebar:
    st.title("Config")
    if "GOOGLE_API_KEY" not in st.secrets:
        key = st.text_input("Enter Google API Key:", type="password")
        if key: st.session_state["google_api_key"] = key
    else:
        st.success("Key Loaded from Secrets!")

st.title("🩺 VaidyaMitra AI")
user_input = st.text_area("మీ లక్షణాలను ఇక్కడ నమోదు చేయండి:")

if st.button("Analyze"):
    if user_input:
        with st.spinner("AI Agents are analyzing..."):
            try:
                final = app_graph.invoke({"symptoms": user_input})
                st.markdown("### Report")
                st.write(final['analysis'])
                if final['urgency'] == "Critical":
                    st.error("🚨 EMERGENCY: Consult a doctor immediately!")
                else:
                    st.success("✅ Condition seems stable.")
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
