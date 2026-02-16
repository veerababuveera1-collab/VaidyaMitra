import streamlit as st
from openai import OpenAI
import os

# --- UI Config ---
st.set_page_config(page_title="VaidyaMitra AI", page_icon="🩺", layout="centered")

# Custom CSS for a medical look
st.markdown("""
    <style>
    .stApp { background-color: #f4f7f6; }
    .report-card { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 6px solid #2ecc71; }
    .stButton>button { background-color: #2ecc71; color: white; font-weight: bold; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Grok Client Setup ---
def get_grok_client():
    # Priority: Streamlit Secrets or Session State
    api_key = st.secrets.get("XAI_API_KEY") or st.session_state.get("grok_api_key")
    if not api_key:
        return None
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"  # Grok API endpoint
    )

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("🏥 Configuration")
    if "XAI_API_KEY" not in st.secrets:
        user_key = st.text_input("Enter Grok API Key (xai-...):", type="password")
        if user_key:
            st.session_state["grok_api_key"] = user_key
            st.success("API Key saved! ✅")
    else:
        st.success("API Key loaded from Secrets! ✅")
    
    st.divider()
    st.markdown("### About VaidyaMitra\nThis AI uses Grok-2 to analyze symptoms and provide preliminary health insights.")

# --- Main App Logic ---
st.title("🩺 VaidyaMitra AI")
st.subheader("Smart Medical Symptom Analyzer")

user_input = st.text_area(
    "మీ లక్షణాలను ఇక్కడ నమోదు చేయండి (Describe your symptoms):",
    placeholder="ఉదా: నాకు మూడు రోజులుగా జ్వరం మరియు ఒళ్ళు నొప్పులు ఉన్నాయి...",
    height=150
)

if st.button("Analyze Symptoms"):
    client = get_grok_client()
    
    if not client:
        st.error("Missing API Key! Please add it in the sidebar.")
    elif not user_input:
        st.warning("Please describe your symptoms before analyzing.")
    else:
        with st.spinner("Grok AI is analyzing your symptoms..."):
            try:
                # Optimized Medical Prompt
                response = client.chat.completions.create(
                    model="grok-2-latest",  # You can also use "grok-beta"
                    messages=[
                        {"role": "system", "content": "You are a highly qualified medical diagnostic assistant. Provide 3 possible conditions, an urgency rating (Low, Medium, Critical), and clear next steps. Always include a disclaimer that this is not a final medical diagnosis."},
                        {"role": "user", "content": f"Symptoms description: {user_input}"}
                    ],
                    temperature=0.3
                )
                
                analysis = response.choices[0].message.content
                
                # Display Results
                st.markdown("### 📋 Analysis Report")
                st.markdown(f"<div class='report-card'>{analysis}</div>", unsafe_allow_html=True)
                
                st.info("⚠️ **Disclaimer:** This analysis is AI-generated for informational purposes only. In case of emergency, please visit a hospital immediately.")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

# --- Footer ---
st.divider()
st.caption("Powered by Grok (xAI) | Developed for VaidyaMitra")
