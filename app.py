import os
import cv2
import streamlit as st
import numpy as np
import PIL.Image
from google.generativeai import configure, GenerativeModel
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO

# Streamlit UI Configuration
st.set_page_config(page_title="Medical Imaging AI", layout="centered")

st.title("\U0001F3E5 Medical Imaging Diagnosis Agent")
st.write("Upload a medical image for professional analysis")

# Sidebar API Key Input
with st.sidebar:
    st.title("ℹ️ Configuration")
    api_key = st.text_input("Enter your Google API Key:", type="password")
    if api_key:
        configure(api_key=api_key)
        st.session_state.api_key = api_key
        st.success("API Key saved!")
    st.info("This AI tool analyzes medical images for diagnostic insights.")
    st.warning("⚠ DISCLAIMER: This is not a medical device. Consult a doctor.")

# Navigation Bar
mode = st.sidebar.radio("Choose Analysis Mode:", ["YOLO Model", "Roboflow Model"])

# Load Models
yolo_model = YOLO("/content/best.pt")  # Load trained YOLO model
roboflow_client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="U5PJXKOLhq73R74qtvwx")

models = {
    "Brain Tumor": "brain-tumour-detection-mri/1",
    "Bone Fracture": "bone-fracture-7fylg/1",
    "Broken Area": "broken-areas-of-body/1"
}

def process_yolo(image, model):
    results = model(image)
    class_names = ["Abrasions", "Bruises", "Burns", "Cut", "Ingrown_Nails", "Laceration", "Stab_Wound"]

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        
        # Draw bounding boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            wound_name = class_names[int(class_ids[i])]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green bounding box
            cv2.putText(image, wound_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Cropping only the first detected wound
        x1, y1, x2, y2 = map(int, boxes[0])
        cropped_wound = image[y1:y2, x1:x2]
        return image, wound_name, cropped_wound

    return image, "No wound detected", None



def process_roboflow(image_path, client):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    detected_conditions = []
    
    for label, model_id in models.items():
        result = client.infer(image_path, model_id=model_id)
        if result.get('predictions', []):
            for obj in result['predictions']:
                x, y, w, h = map(int, [obj['x'], obj['y'], obj['width'], obj['height']])
                x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
                cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(original_image, f"{label} ({obj['confidence']:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                detected_conditions.append(label)
    return original_image, detected_conditions if detected_conditions else ["No anomalies detected"]

uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image, caption="Uploaded Medical Image", use_container_width=True)
    
    analyze_button = st.button("🔍 Analyze Image")
    
    if analyze_button:
        with st.spinner("🔄 Analyzing image... Please wait."):
            try:
                if mode == "YOLO Model":
                    yolo_annotated, yolo_diagnosis, cropped_wound = process_yolo(image_np.copy(), yolo_model)
                    st.image(yolo_annotated, caption=f"YOLO Detection: {yolo_diagnosis}", use_column_width=True)
                    if cropped_wound is not None:
                        st.image(cropped_wound, caption="Cropped Wound Image for Gemini", use_column_width=True)
                        query = f"""
                        The following image is a cropped wound detected by an AI model.
                        The YOLO wound detection model identified it as: {yolo_diagnosis}.
                        Please analyze and provide a medical report.
                        """
                    else:
                        query = "No wound detected in the image."
                else:
                    temp_image_path = "temp_image.jpg"
                    cv2.imwrite(temp_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                    roboflow_annotated, roboflow_diagnoses = process_roboflow(temp_image_path, roboflow_client)
                    st.image(roboflow_annotated, caption=f"Roboflow Detection: {', '.join(roboflow_diagnoses)}", use_column_width=True)
                    query = f"""
                    The following image contains detected medical conditions:
                    - Roboflow Detection: {', '.join(roboflow_diagnoses)}
                    You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
                    """
                
                gemini_model = GenerativeModel("gemini-2.0-flash")
                response = gemini_model.generate_content(query)
                st.markdown("### 📋 AI Analysis Report")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Analysis error: {e}")


