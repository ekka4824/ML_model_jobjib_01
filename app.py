import streamlit as st
import pandas as pd
import joblib
import json
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Workpiece Clustering App", layout="wide")

CONFIG_PATH = "config/features_config.json"
MODEL_PATH = "models/kmeans_k3_model.joblib"

# โหลดรายชื่อตัวแปร (ไม่ cache เพื่อให้แก้ไขไฟล์แล้วมีผลทันทีเมื่อ refresh)
def load_features():
    if not os.path.exists(CONFIG_PATH):
        st.error(f"No config file found at : {CONFIG_PATH}")
        st.stop()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("features", [])

# โหลดโมเดล (ใช้ cache เพื่อความเร็ว ไม่ต้องโหลดใหม่ทุกครั้งที่กดปุ่ม)
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"No ML model found at : {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

features = load_features()
model = load_model()

st.title("Workpiece Clustering for Die Casting Part (Partname : XXX) (kmean)")
st.caption(f"Load information of Parameter {len(features)}")

# สร้าง Tabs สำหรับ 2 รูปแบบ Input
tab_manual, tab_csv = st.tabs(["Manual Input", "Upload CSV"])

# ==========================================
# TAB 1: Manual Input
# ==========================================
with tab_manual:
    st.subheader("Input Parameters value")
    with st.form("manual_input_form"):
        input_data = {}
        # แบ่งการแสดงผลเป็น 4 คอลัมน์ เพื่อความสวยงามและกรอกง่าย
        cols = st.columns(4)
        for idx, feature_name in enumerate(features):
            col = cols[idx % 4]
            with col:
                input_data[feature_name] = st.number_input(
                    label=feature_name,
                    value=0.0,
                    format="%.4f",
                    key=f"num_{idx}"
                )
        
        submitted = st.form_submit_button("Predict Cluster", type="primary")
        
        if submitted:
            df_input = pd.DataFrame([input_data])
            # จัดเรียงคอลัมน์ให้ตรงกับลำดับใน features.json เสมอ
            df_input = df_input[features]
            try:
                prediction = model.predict(df_input)
                st.success(f"Results of prediction : **Cluster {prediction[0]}**")
            except Exception as e:
                st.error(f"Error found : {e}")

# ==========================================
# TAB 2: Upload CSV
# ==========================================
with tab_csv:
    st.subheader("Upload CSV for multi cluster")
    
    # ปุ่มดาวน์โหลดไฟล์ตัวอย่าง (Template CSV)
    sample_df = pd.DataFrame(columns=features)
    csv_template = sample_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download CSV Template",
        data=csv_template,
        file_name="template_37_features.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Example of upload data (5 row):")
            st.dataframe(df_upload.head())
            
            # ตรวจสอบว่ามีคอลัมน์ครบตาม features.json หรือไม่
            missing_cols = [col for col in features if col not in df_upload.columns]
            
            if missing_cols:
                st.error(f"File CSV on columns: {', '.join(missing_cols)}")
            else:
                if st.button("Process CSV file", type="primary"):
                    # ดึงเฉพาะ 37 คอลัมน์ตามลำดับที่กำหนด เพื่อป้องกันคอลัมน์เกินหรือสลับตำแหน่ง
                    X_predict = df_upload[features]
                    
                    # ตรวจสอบค่าว่าง (NaN)
                    if X_predict.isnull().any().any():
                        st.warning("File has Null/NaN , Results may be not correct or error")
                    
                    predictions = model.predict(X_predict)
                    df_result = df_upload.copy()
                    df_result["Predicted_Cluster"] = predictions
                    
                    st.success("Calculation successfully!")
                    st.dataframe(df_result)
                    
                    # สรุปจำนวนแต่ละ Cluster
                    st.write("Summary of workpiece in group:")
                    st.bar_chart(df_result["Predicted_Cluster"].value_counts())
                    
                    # ปุ่มดาวน์โหลดผลลัพธ์
                    result_csv = df_result.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="download CSV file of clustering",
                        data=result_csv,
                        file_name="clustering_results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Cannot read CSV file : {e}")