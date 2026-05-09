import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Lấy đường dẫn của thư mục chứa file 
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "model") 

@st.cache_resource
def load_assets():
    """Tải model và các công cụ tiền xử lý một lần duy nhất"""
    model = joblib.load(os.path.join(MODEL_PATH, 'best_logistic_regression.pkl'))
    scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
    imputer = joblib.load(os.path.join(MODEL_PATH, 'imputer.pkl'))
    threshold = joblib.load(os.path.join(MODEL_PATH, 'optimal_threshold.pkl'))
    return model, scaler, imputer, threshold

# Kiểm tra sự tồn tại của thư mục model 
if not os.path.exists(MODEL_PATH):
    st.error(f" Không tìm thấy thư mục 'model' tại: {MODEL_PATH}")
    st.info("Vui lòng đảm bảo bạn đã tạo thư mục 'model' và để các file .pkl vào đó.")
    st.stop()

try:
    model, scaler, imputer, optimal_threshold = load_assets()
except Exception as e:
    st.error(f" Lỗi khi tải file mô hình: {e}")
    st.stop()

# 2. GIAO DIỆN STREAMLIT
st.set_page_config(page_title="Dự đoán Đột quỵ", page_icon="🛡️", layout="wide")

st.title("🛡️ Hệ Thống Dự Đoán Nguy Cơ Đột Quỵ")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Thông tin cá nhân")
    age = st.number_input("Tuổi", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
    ever_married = st.selectbox("Đã từng kết hôn?", ["Rồi", "Chưa"])
    work_type = st.selectbox("Loại hình công việc", ["Tư nhân", "Tự kinh doanh", "Việc làm nhà nước", "con cái", "Chưa từng làm việc"])
    residence_type = st.selectbox("Khu vực sinh sống", ["Đô thị", "Nông thôn"])

with col2:
    st.subheader("Chỉ số sức khỏe")
    hypertension = st.selectbox("Bị cao huyết áp?", ["Không", "Có"])
    heart_disease = st.selectbox("Bị bệnh tim?", ["Không", "Có"])
    avg_glucose_level = st.number_input("Chỉ số Glucose trung bình (mg/dL)", min_value=50.0, max_value=300.0, value=100.0)
    bmi = st.number_input("Chỉ số BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoking_status = st.selectbox("Tình trạng hút thuốc", ["Trước đây từng hút thuốc", "Chưa bao giờ hút thuốc", "Hút thuốc", "Không rõ"])

# 3. XỬ LÝ DỰ ĐOÁN
st.markdown("---")
if st.button("PHÂN TÍCH NGUY CƠ", use_container_width=True):
    # Tạo DataFrame khớp chính xác với các cột mà mô hình đã được huấn luyện
    input_dict = {
        'age': age,
        'hypertension': 1 if hypertension == "Có" else 0,
        'heart_disease': 1 if heart_disease == "Có" else 0,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'gender_Male': 1 if gender == "Nam" else 0,
        'ever_married_Yes': 1 if ever_married == "Rồi" else 0,
        'work_type_Never_worked': 1 if work_type == "Never_worked" else 0,
        'work_type_Private': 1 if work_type == "Private" else 0,
        'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
        'work_type_children': 1 if work_type == "children" else 0,
        'Residence_type_Urban': 1 if residence_type == "Urban" else 0,
        'smoking_status_formerly smoked': 1 if smoking_status == "formerly smoked" else 0,
        'smoking_status_never smoked': 1 if smoking_status == "never smoked" else 0,
        'smoking_status_smokes': 1 if smoking_status == "smokes" else 0
    }
    
    input_data = pd.DataFrame([input_dict])

    # Tiền xử lý dữ liệu
    # 1. Imputer (Xử lý giá trị thiếu cho BMI)
    input_data[['bmi']] = imputer.transform(input_data[['bmi']])
    
    # 2. Scaler (Chuẩn hóa các biến số)
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Dự đoán xác suất
    prob = model.predict_proba(input_data)[0][1]
    
    # So sánh với ngưỡng tối ưu để đưa ra dự đoán cuối cùng
    thresh = optimal_threshold[0] if isinstance(optimal_threshold, (list, np.ndarray)) else optimal_threshold
    prediction = 1 if prob >= thresh else 0

    # Hiển thị kết quả
    st.subheader("Kết quả phân tích")
    risk_percent = prob * 100

    if prediction == 1:
        st.error(f"⚠️ **CẢNH BÁO:** Bệnh nhân có nguy cơ đột quỵ cao!")
    else:
        st.success(f"✅ **AN TÂM:** Nguy cơ đột quỵ hiện tại ở mức thấp.")

    # Hiển thị thông số chi tiết
    c1, c2 = st.columns(2)
    c1.metric("Xác suất dự đoán", f"{risk_percent:.2f}%")
    c2.metric("Ngưỡng cảnh báo", f"{thresh:.2f}")
    
    st.info("Kết quả này chỉ mang tính chất tham khảo dựa trên mô hình học máy và không thay thế cho chẩn đoán y tế chuyên nghiệp. Vui lòng tham khảo ý kiến bác sĩ để được tư vấn chính xác.")