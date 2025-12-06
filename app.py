# app.py
import streamlit as st
import pandas as pd
import joblib

# 1. Load mô hình đã huấn luyện
try:
    model = joblib.load('house_model.pkl')
except:
    st.error("Chưa tìm thấy file mô hình 'house_model.pkl'. Hãy chạy file train_model.py trước!")
    st.stop()

# 2. Tạo giao diện Web
st.title("🏡 Hệ Thống Dự Đoán Giá Nhà Việt Nam")
st.write("Nhập thông tin căn nhà để dự đoán giá thị trường.")

# Tạo form nhập liệu (Input)
# Layout chia làm 2 cột cho đẹp
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Diện tích (m2)", min_value=10.0, value=50.0)
    floors = st.number_input("Số tầng", min_value=1.0, value=2.0)
    bedrooms = st.number_input("Số phòng ngủ", min_value=1.0, value=2.0)

with col2:
    bathrooms = st.number_input("Số phòng vệ sinh", min_value=1.0, value=2.0)
    frontage = st.number_input("Mặt tiền (m)", min_value=0.0, value=5.0)
    access_road = st.number_input("Đường vào (m)", min_value=0.0, value=5.0)

# 3. Nút dự đoán và Xử lý
if st.button("🔍 Dự đoán giá ngay", type="primary"):
    # Tạo dataframe từ dữ liệu nhập vào (đúng thứ tự features lúc train)
    input_data = pd.DataFrame([[area, floors, bedrooms, bathrooms, frontage, access_road]], 
                              columns=['Area', 'Floors', 'Bedrooms', 'Bathrooms', 'Frontage', 'Access Road'])
    
    # Thực hiện dự đoán
    prediction = model.predict(input_data)
    
    # Hiển thị kết quả
    st.success(f"💰 Giá nhà dự đoán: **{prediction[0]:.2f} Tỷ VNĐ**")
    
    # (Optional) Hiển thị thêm thông tin vui
    if prediction[0] > 10:
        st.balloons()