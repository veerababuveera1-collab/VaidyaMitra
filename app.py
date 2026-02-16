import streamlit as st
import google.generativeai as genai

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VaidyaMitra AI - Gemini Edition",
    page_icon="🩺",
    layout="centered"
)

# --- 2. CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { border-radius: 12px; border: 2px solid #4285F4; }
    .stButton>button { 
        width: 100%; 
        border-radius: 12px; 
        background-color: #4285F4; 
        color: white; 
        font-weight: bold;
        height: 3.5em;
    }
    .result-card { 
        background-color: white; 
        padding: 25px; 
        border-radius: 15px; 
        border-left: 8px solid #4285F4;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. GEMINI MODEL SETUP ---
def setup_gemini():
    # Streamlit Secrets లో 'GOOGLE_API_KEY' ఉండాలి
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("temp_google_key")
    
    if not api_key:
        return None
        
    try:
        genai.configure(api_key=api_key)
        # 'models/' prefix వాడటం వల్ల 404 Error నివారించబడుతుంది
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Configuration Error: {e}")
        return None

# --- 4. SIDEBAR SETTINGS ---
with st.sidebar:
    st.title("🔐 Connection")
    
    if "GOOGLE_API_KEY" in st.secrets:
        st.success("API Key Loaded from Secrets! ✅")
        api_ready = True
    else:
        g_key = st.text_input("Enter Google API Key:", type="password")
        if g_key:
            st.session_state["temp_google_key"] = g_key
            st.success("Key Accepted! ✅")
            api_ready = True
        else:
            st.warning("Please enter your API Key.")
            api_ready = False

    st.divider()
    st.info("VaidyaMitra AI అనేది Google Gemini 1.5 Flash మోడల్‌ను ఉపయోగించే ఉచిత వెర్షన్.")

# --- 5. MAIN INTERFACE ---
st.title("🩺 VaidyaMitra AI")
st.markdown("##### మీ ఆరోగ్య సమస్యలను సులభంగా విశ్లేషించండి")

user_input = st.text_area(
    "మీ లక్షణాలను ఇక్కడ వివరించండి (Describe symptoms):", 
    placeholder="ఉదా: నాకు రెండు రోజులుగా జ్వరం మరియు దగ్గు ఉంది...",
    height=150
)

# --- 6. ANALYSIS LOGIC ---
if st.button("విశ్లేషించు (Analyze Now)"):
    if not api_ready:
        st.error("ముందుగా సైడ్‌బార్‌లో API Key ని నమోదు చేయండి!")
    elif not user_input:
        st.warning("దయచేసి మీ లక్షణాలను టైప్ చేయండి.")
    else:
        model = setup_gemini()
        if model:
            with st.spinner("Gemini AI విశ్లేషిస్తోంది, దయచేసి వేచి ఉండండి..."):
                try:
                    # Professional Health Prompt
                    prompt = f"""
                    You are VaidyaMitra, a medical assistant AI. Analyze these symptoms: "{user_input}"
                    Provide the following in both Telugu and English:
                    1. 3 Potential causes.
                    2. Urgency level (Low, Medium, or High).
                    3. Recommended next steps.
                    Add a strict medical disclaimer at the end.
                    """
                    
                    response = model.generate_content(prompt)
                    
                    # Displaying Output
                    st.subheader("📋 విశ్లేషణ నివేదిక (Analysis Report)")
                    st.markdown(f"<div class='result-card'>{response.text}</div>", unsafe_allow_html=True)
                    
                    st.divider()
                    st.warning("⚠️ **గమనిక:** ఇది AI అందించిన సమాచారం మాత్రమే. ఖచ్చితమైన చికిత్స కోసం వైద్యుడిని సంప్రదించండి.")
                    
                except Exception as e:
                    st.error(f"Error: {e}")

# --- 7. FOOTER ---
st.divider()
st.caption("Powered by Google Gemini 1.5 Flash | Built for VaidyaMitra")
