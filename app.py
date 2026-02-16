import streamlit as st
import google.generativeai as genai

# --- UI Config ---
st.set_page_config(page_title="VaidyaMitra AI", page_icon="🩺", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    .report-box { background-color: #f9f9f9; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- API Setup ---
def setup_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        # Using the most stable model name
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"API Configuration Error: {e}")
        return None

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.info("ఈ యాప్ మీ లక్షణాలను విశ్లేషించడానికి Google Gemini AI ని ఉపయోగిస్తుంది.")
    
    # Priority: Secrets > Manual Input
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.session_state.get("manual_api_key")
    
    if not api_key:
        manual_key = st.text_input("Google API Key ఎంటర్ చేయండి:", type="password")
        if manual_key:
            st.session_state["manual_api_key"] = manual_key
            st.rerun()
    else:
        st.success("API Key సిద్ధంగా ఉంది! ✅")
        if st.button("Clear Key"):
            st.session_state["manual_api_key"] = None
            st.rerun()

# --- Main UI ---
st.title("🩺 VaidyaMitra AI")
st.write("మీ ఆరోగ్య సమస్యలను కింద వివరించండి (English or తెలుగు).")

user_input = st.text_area("లక్షణాలు (Symptoms):", placeholder="ఉదా: నాకు రెండు రోజులుగా జ్వరం మరియు తలనొప్పి ఉంది...", height=150)

if st.button("విశ్లేషించు (Analyze)"):
    if not api_key:
        st.warning("దయచేసి సైడ్‌బార్‌లో API Key ని నమోదు చేయండి.")
    elif not user_input:
        st.warning("ముందుగా మీ లక్షణాలను టైప్ చేయండి.")
    else:
        model = setup_gemini(api_key)
        if model:
            with st.spinner("AI విశ్లేషిస్తోంది, దయచేసి వేచి ఉండండి..."):
                try:
                    # Professional Medical Prompt
                    prompt = f"""
                    You are a professional medical assistant. Analyze the following symptoms:
                    "{user_input}"
                    
                    Please provide:
                    1. 3 Potential medical conditions.
                    2. Urgency Level (LOW, MEDIUM, or CRITICAL).
                    3. Recommended next steps.
                    
                    Important: Mention that this is an AI-generated report and not a substitute for professional medical advice.
                    """
                    
                    response = model.generate_content(prompt)
                    
                    st.subheader("📋 విశ్లేషణ నివేదిక (Analysis Report)")
                    st.markdown(f"<div class='report-box'>{response.text}</div>", unsafe_allow_html=True)
                    
                    st.divider()
                    st.caption("⚠️ గమనిక: ఇది కేవలం సమాచారం కోసం మాత్రమే. అత్యవసర స్థితిలో వెంటనే వైద్యుడిని సంప్రదించండి.")
                    
                except Exception as e:
                    if "404" in str(e):
                        st.error("Error: మోడల్ 'gemini-1.5-flash' కనుగొనబడలేదు. మీ API Key కి ఈ మోడల్ పర్మిషన్ ఉందో లేదో చెక్ చేయండి.")
                    else:
                        st.error(f"Error: {e}")

# --- Footer ---
st.markdown("---")
st.center = st.caption("Powered by Google Gemini AI")
