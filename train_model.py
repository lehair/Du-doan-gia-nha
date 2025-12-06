# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. TẢI DỮ LIỆU
# ==========================================
print("dang tai du lieu...")
try:
    df = pd.read_csv('vietnam_housing_dataset.csv')
except FileNotFoundError:
    print("Loi: Khong tim thay file 'vietnam_housing_dataset.csv'")
    exit()

# ==========================================
# 2. TIỀN XỬ LÝ (Cleaning)
# ==========================================
print("Dang tien xu ly...")

# Chọn các đặc trưng (Features) số quan trọng
features = ['Area', 'Floors', 'Bedrooms', 'Bathrooms', 'Frontage', 'Access Road']
target = 'Price'

# Xử lý giá trị thiếu (Missing Values)
# Với dữ liệu số, ta điền bằng giá trị trung vị (median) để tránh sai lệch do nhiễu
for col in features:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Xóa các dòng mà Giá nhà (Target) bị rỗng
df = df.dropna(subset=[target])

# ==========================================
# 3. XỬ LÝ DỮ LIỆU (Feature Engineering)
# ==========================================
# (Ở bài toán đơn giản này, bước 2 và 3 thường gộp chung. 
# Nếu dữ liệu phức tạp hơn, bước này sẽ bao gồm: chuẩn hóa dữ liệu, one-hot encoding...)

X = df[features]
y = df[target]

# ==========================================
# 4. HUẤN LUYỆN MÔ HÌNH
# ==========================================
print("Dang huan luyen mo hinh...")
# Chia dữ liệu: 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và train
model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# 5. ĐÁNH GIÁ MÔ HÌNH & LƯU
# ==========================================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")
print("-" * 30)

# Lưu mô hình đã train ra file .pkl để App có thể dùng
joblib.dump(model, 'house_model.pkl')
print("Da luu mo hinh thanh cong vao file 'house_model.pkl'")