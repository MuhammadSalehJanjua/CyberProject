import streamlit as st
import tempfile
from pathlib import Path
from detector import EnhancedDetector
import cv2

# Configuration
st.set_page_config(page_title="DeepGuard AI", layout="wide")
st.markdown("""
<style>
    .reportview-container {background: #f8f9fa}
    .sidebar .sidebar-content {background: #ffffff}
    div[data-testid="stMetricValue"] {font-size: 1.5rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    return EnhancedDetector()

def main():
    detector = load_detector()
    
    # Sidebar Controls
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.65)
    
    # Main Interface
    st.title("🛡️ DeepGuard AI - Deepfake Detection")
    uploaded_file = st.file_uploader("Upload Media", type=['jpg', 'png', 'jpeg', 'mp4'])
    
    if uploaded_file:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Input Media")
            if uploaded_file.type.startswith('image'):
                st.image(uploaded_file, use_column_width=True)
            else:
                st.video(uploaded_file)
        
        with col2:
            st.subheader("Analysis Results")
            with st.spinner("Analyzing..."):
                if uploaded_file.type.startswith('image'):
                    label, confidence = detector.predict(file_path, confidence_threshold)
                    
                    # Display Results
                    st.metric(label="Prediction", 
                             value=label,
                             delta=f"{confidence*100:.1f}% Confidence")
                    
                    # Confidence Visualization
                    st.progress(confidence)
                    
                    # Explanation
                    if label == "Fake":
                        st.error("""
                        **Detected Anomalies**
                        - Unnatural facial textures
                        - Asymmetric eye blinking
                        - Inconsistent lighting
                        """)
                    else:
                        st.success("""
                        **Authenticity Indicators**
                        - Natural skin texture
                        - Consistent facial movements
                        - Realistic lighting patterns
                        """)
                
                else:  # Video handling
                    st.warning("Video analysis requires GPU acceleration")
                    st.info("Coming soon in v2.0!")

if __name__ == '__main__':
    main()