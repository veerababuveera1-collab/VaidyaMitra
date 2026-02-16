import streamlit as st
from openai import OpenAI

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VaidyaMitra AI - Grok Edition",
    page_icon="🩺",
    layout="centered"
)

# --- 2. CUSTOM STYLING (Medical Theme) ---
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .stTextArea textarea { border-radius: 12px; border: 2px solid #3498db; }
    .stButton>button { 
        width: 100%; 
        border-radius: 12px; 
        background-color: #27ae60; 
        color: white; 
        font-weight: bold;
        font-size: 18px;
        height: 3.5em;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #2ecc71; border: none; }
    .result-box { 
        background-color: white; 
        padding: 25px; 
        border-radius: 15px; 
        border-left: 8px solid #27ae60;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. GROK CLIENT SETUP ---
def get_grok_client():
    # Streamlit Secrets (XAI_API_KEY) నుండి కీని తీసుకుంటుంది
    api_key = st.secrets.get("XAI_API_KEY") or st.session_state.get("grok_api_key")
    
    if not api_key:
        return None
        
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )

# --- 4. SIDEBAR & KEY CHECK ---
with st.sidebar:
    st.title("🛡️ Connection Status")
    
    # Secrets లో కీ ఉందో లేదో తనిఖీ చేస్తుంది
    if "XAI_API_KEY" in st.secrets:
        st.success("API Key loaded from Settings! ✅")
        api_ready = True
    elif "grok_api_key" in st.session_state:
        st.success("Manual Key Loaded! ✅")
        api_ready = True
    else:
        manual_key = st.text_input("Enter Grok API Key (xai-...):", type="password")
        if manual_key:
            st.session_state["grok_api_key"] = manual_key
            st.rerun()
        else:
            st.warning("Please add API Key in Settings or here.")
            api_ready = False

    st.divider()
    st.info("VaidyaMitra AI మీ ఆరోగ్య లక్షణాలను విశ్లేషించి ప్రాథమిక అవగాహన కల్పిస్తుంది.")

# --- 5. MAIN INTERFACE ---
st.title("🩺 VaidyaMitra AI")
st.markdown("##### మీ లక్షణాలను విశ్లేషించే స్మార్ట్ మెడికల్ అసిస్టెంట్")

user_input = st.text_area(
    "మీ ఆరోగ్య సమస్యలను ఇక్కడ వివరించండి (Describe symptoms):", 
    placeholder="ఉదా: నాకు రెండు రోజులుగా విపరీతమైన తలనొప్పి మరియు జ్వరం ఉంది...",
    height=150
)

# --- 6. ANALYSIS LOGIC ---
if st.button("విశ్లేషించు (Analyze Symptoms)"):
    if not api_ready:
        st.error("ముందుగా API Key ని నమోదు చేయండి!")
    elif not user_input:
        st.warning("దయచేసి మీ లక్షణాలను బాక్స్‌లో టైప్ చేయండి.")
    else:
        client = get_grok_client()
        if client:
            with st.spinner("Grok AI మీ లక్షణాలను విశ్లేషిస్తోంది, దయచేసి వేచి ఉండండి..."):
                try:
                    # 'grok-beta' is used to avoid 'Model Not Found' errors
                    response = client.chat.completions.create(
                        model="grok-beta", 
                        messages=[
                            {"role": "system", "content": "You are VaidyaMitra, a professional medical assistant. Analyze the user's symptoms and provide a response in both Telugu and English. Structure the response with: 1. Potential Causes, 2. Urgency Level, 3. Suggested Next Steps. Always include a clear disclaimer that you are an AI, not a doctor."},
                            {"role": "user", "content": user_input}
                        ],
                        temperature=0.3
                    )
                    
                    # Displaying Output
                    result = response.choices[0].message.content
                    st.subheader("📋 విశ్లేషణ నివేదిక (Analysis Report)")
                    st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
                    
                    st.divider()
                    st.warning("⚠️ **గమనిక:** ఇది కేవలం సమాచారం కోసం మాత్రమే. మీకు అత్యవసరమైతే వెంటనే సమీపంలోని వైద్యుడిని లేదా ఆసుపత్రిని సంప్రదించండి.")
                    
                except Exception as e:
                    if "400" in str(e):
                        st.error("API Error: మోడల్ కనుగొనబడలేదు లేదా కీ చెల్లదు. దయచేసి మీ Grok క్రెడిట్స్ మరియు కీని తనిఖీ చేయండి.")
                    else:
                        st.error(f"Error: {e}")

# --- 7. FOOTER ---
st.divider()
st.caption("Powered by xAI Grok-Beta | Built for VaidyaMitra")
