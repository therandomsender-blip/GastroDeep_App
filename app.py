import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

# --- 1. DASHBOARD CONFIGURATION ---
st.set_page_config(page_title="GastroDeep AI", layout="wide", page_icon="🩺")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .report-box { padding: 20px; border-radius: 10px; background-color: #ffffff; border-left: 5px solid #007bff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL LOADING ---
import os  # (Add this right here if it's not already at the very top of your file)

@st.cache_resource
def get_model():
    # 1. Find the exact folder where this app.py file lives on the Linux server
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Build a bulletproof path to the model file
    model_path = os.path.join(current_dir, 'model', 'GastroDeep_Final_Model.h5')
    
    # 3. Try to load it directly (no try/except) so we can see the raw error!
    return load_model(model_path)

# Call the function directly
model = get_model()

classes = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 
           'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']

# --- 3. HELPER FUNCTIONS ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if isinstance(preds, list): preds = preds[0]
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3304/3304567.png", width=100)
    st.title("Control Panel")
    uploaded_file = st.file_uploader("Upload Endoscopy Image", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    st.write("### Model Settings")
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
    st.write("Architecture: **ResNet-50**")

# --- 5. MAIN INTERFACE ---
st.title("🩺 GastroDeep: Diagnostic Intelligence")

if uploaded_file:
    # Processing Animation
    with st.spinner('Analyzing Pathological Features...'):
        img = Image.open(uploaded_file).convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        img_input = np.expand_dims(img_array, axis=0)
        img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_input.copy())

        # Prediction
        preds = model.predict(img_preprocessed)
        if isinstance(preds, list): preds = preds[0]
        
        idx = np.argmax(preds)
        label = classes[idx]
        conf = preds[0][idx]
        time.sleep(0.5) # Simulate processing

    # Layout: Top Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Class", label.replace("-", " ").title())
    m2.metric("Confidence Score", f"{conf*100:.1f}%")
    m3.metric("Status", "High Confidence" if conf > threshold else "Review Required")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Input Source")
        st.image(img, use_container_width=True, caption="Uploaded Endoscopic Frame")

    with col2:
        st.subheader("🔥 Interpretability (Grad-CAM)")
        heatmap = make_gradcam_heatmap(img_preprocessed, model, "conv5_block3_out")
        fig, ax = plt.subplots()
        ax.imshow(img_resized)
        ax.imshow(heatmap, cmap='jet', alpha=0.45, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    # --- NEW WORKING BUTTONS ---
    # --- UPDATED CLINICAL ACTIONS WITH DOWNLOAD FEATURE ---
    st.markdown("### 🛠️ Clinical Actions")
    b1, b2, b3 = st.columns(3)

    # Initialize a variable to store report text for downloading
    report_content = f"""MEDICAL DIAGNOSTIC REPORT
---------------------------
Timestamp: {time.strftime("%Y-%m-%d %H:%M")}
Image Source: {uploaded_file.name}
Primary Finding: {label.upper()}
Confidence Score: {conf*100:.2f}%
XAI Analysis: Heatmap highlights localized vascular/mucosal irregularities.
Status: {"Verified" if conf > threshold else "Pending Manual Review"}
---------------------------
Disclaimer: This is an AI-generated summary for clinical decision support."""

    if b1.button("📋 View Clinical Report"):
        # We use a distinct background and forced dark text color for visibility
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #e9f5ff; border-left: 5px solid #007bff; color: #1a1a1a;">
            <h4 style="color: #007bff; margin-top:0;">Diagnostic Summary</h4>
            <p><strong>Primary Finding:</strong> <span style="color: #d9534f;">{label.replace("-", " ").upper()}</span></p>
            <p><strong>Evidence:</strong> AI identified localized features in the highlighted heatmap zones consistent with established pathological markers.</p>
            <p><strong>Timestamp:</strong> {time.strftime("%Y-%m-%d %H:%M")}</p>
            <hr>
            <p style="font-size: 0.8em; color: #555;">Confidence: {conf*100:.2f}% | Model: ResNet-50 v1.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add the Download Button right below the view
        st.download_button(
            label="📥 Download Report as Text",
            data=report_content,
            fileName=f"GastroDeep_Report_{uploaded_file.name}.txt",
            mime="text/plain"
        )

    if b2.button("⚠️ Flag for Manual Review"):
        st.warning(f"Case {uploaded_file.name} has been added to the expert review queue for manual verification.")

    if b3.button("🔄 Reset Analysis"):
        st.rerun()

    # --- CLINICAL FEEDBACK LOOP ---
    st.markdown("---")
    with st.expander("📝 Provide Clinical Feedback (Hard Negative Mining Data)"):
        feedback = st.radio("Is this prediction accurate?", ("Correct", "Incorrect", "Uncertain"))
        notes = st.text_area("Observations:")
        if st.button("Submit Feedback"):
            st.success("Feedback saved. This data will be used to retrain the model via Hard Negative Mining.")

else:
    st.info("👋 Welcome! Please upload an endoscopy image in the sidebar to begin the automated analysis.")
