import streamlit as st
from openai import OpenAI

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VaidyaMitra AI - Grok Edition",
    page_icon="🩺",
    layout="centered"
)

# --- 2. CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stTextArea textarea { border-radius: 10px; border: 1px solid #2ecc71; }
    .stButton>button { 
        width: 100%; 
        border-radius: 10px; 
        background-color: #2ecc71; 
        color: white; 
        font-weight: bold;
        height: 3em;
    }
    .result-box { 
        background-color: white; 
        padding: 20px; 
        border-radius: 15px; 
        border-left: 6px solid #2ecc71;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. GROK CLIENT SETUP ---
def get_grok_client():
    # Priority 1: Streamlit Secrets (for production)
    # Priority 2: Session State (for manual testing)
    api_key = st.secrets.get("XAI_API_KEY") or st.session_state.get("grok_api_key")
    
    if not api_key:
        return None
        
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )

# --- 4. SIDEBAR SETTINGS ---
with st.sidebar:
    st.title("🛡️ Secure Access")
    st.write("VaidyaMitra AI ని ఉపయోగించడానికి Grok API కీ అవసరం.")
    
    # Check if key is in secrets
    if "XAI_API_KEY" in st.secrets:
        st.success("API Key loaded from Secrets! ✅")
        api_ready = True
    else:
        manual_key = st.text_input("Enter Grok API Key (xai-...):", type="password")
        if manual_key:
            st.session_state["grok_api_key"] = manual_key
            st.success("Key accepted! ✅")
            api_ready = True
        else:
            st.warning("Please enter your key to proceed.")
            api_ready = False

    st.divider()
    st.info("గమనిక: మీ డేటా మరియు కీ భద్రంగా ఉంటాయి. మేము ఎక్కడా స్టోర్ చేయము.")

# --- 5. MAIN INTERFACE ---
st.title("🩺 VaidyaMitra AI")
st.markdown("#### మీ ఆరోగ్య సహకారి (Your AI Health Assistant)")

st.write("కింద ఉన్న బాక్స్‌లో మీ లక్షణాలను (Symptoms) వివరించండి. Grok AI వాటిని విశ్లేషిస్తుంది.")

user_input = st.text_area(
    "Describe symptoms:", 
    placeholder="ఉదా: నాకు రెండు రోజులుగా తలనొప్పి మరియు జ్వరం ఉంది...",
    height=150
)

# --- 6. ANALYSIS LOGIC ---
if st.button("విశ్లేషించు (Analyze Now)"):
    if not api_ready:
        st.error("సైడ్‌బార్‌లో API Key నమోదు చేయండి!")
    elif not user_input:
        st.warning("ముందుగా మీ లక్షణాలను వివరించండి.")
    else:
        client = get_grok_client()
        if client:
            with st.spinner("Grok AI విశ్లేషిస్తోంది..."):
                try:
                    # Professional Prompting
                    response = client.chat.completions.create(
                        model="grok-2-latest", 
                        messages=[
                            {"role": "system", "content": "You are a professional medical assistant named VaidyaMitra. Use Telugu and English for the response. Provide 3 possible causes, urgency level, and immediate steps. Always add a disclaimer: 'This is not a medical diagnosis.'"},
                            {"role": "user", "content": user_input}
                        ],
                        temperature=0.3
                    )
                    
                    # Displaying Output
                    result = response.choices[0].message.content
                    st.subheader("📋 విశ్లేషణ నివేదిక (Analysis Report)")
                    st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
                    
                    st.warning("⚠️ **Disclaimer:** ఇది కేవలం సమాచారం కోసం మాత్రమే. అత్యవసర స్థితిలో వెంటనే వైద్యుడిని సంప్రదించండి.")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("ఒకవేళ 'Model Not Found' అని వస్తే, కోడ్‌లో 'grok-2-latest' ని 'grok-beta' గా మార్చండి.")

# --- 7. FOOTER ---
st.divider()
st.caption("Powered by xAI Grok | Developed for VaidyaMitra")
