import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Object Counter", layout="wide")
MODEL_PATH = "best.pt" # ตรวจสอบให้แน่ใจว่ามีไฟล์นี้ใน GitHub

# --- LOAD MODEL ---
@st.cache_resource
def load_yolo():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

model = load_yolo()

# --- UI SIDEBAR ---
st.sidebar.title("⚙️ Configuration")
operator = st.sidebar.text_input("ผู้ปฏิบัติงาน (Operator)")
lot_number = st.sidebar.text_input("รหัสงาน (Lot Number)")
conf_threshold = st.sidebar.slider("Sensitivity (%)", 0, 100, 25) / 100

# --- MAIN INTERFACE ---
st.title("TK-VISION : SMART OBJECT COUNTER")

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("📊 Dashboard")
    placeholder_count = st.empty()
    if st.button("💾 บันทึกยอดปัจจุบัน"):
        st.success(f"บันทึก Lot {lot_number} เรียบร้อย (จำลอง)")

with col1:
    # เลือกแหล่งที่มาของภาพ (Streamlit รัน Webcam บน Cloud ยากเล็กน้อย มักใช้ Upload)
    img_file = st.file_uploader("อัปโหลดรูปภาพเพื่อตรวจนับ", type=['jpg', 'png', 'jpeg'])

    if img_file is not None:
        image = Image.open(img_file)
        frame = np.array(image)
        
        if model:
            results = model.predict(frame, conf=conf_threshold)
            count = 0
            
            # วาดผลลัพธ์
            for r in results:
                count = len(r.boxes)
                res_plotted = r.plot()
                st.image(res_plotted, caption=f"ตรวจพบทั้งหมด {count} ชิ้น", use_column_width=True)
                placeholder_count.metric("OBJECTS FOUND", count)
        else:
            st.error("ไม่พบไฟล์ Model (best.pt)")