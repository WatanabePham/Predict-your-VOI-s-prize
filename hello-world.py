# == Import necessary libraries ==
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

# == Load and preprocess data ==
def select_features(df, feature_name):
    return df[feature_name].values

def select_label(df, label_name):
    return df[label_name].values


ket_qua_thi_thu = pd.read_csv('ket_qua_thi_thu_utf8.csv')
feature_ket_qua_tt = select_features(ket_qua_thi_thu, ["ANDORXOR", "CROSS", "SLAMP", "division", "sort", "zip", "Points"])
target_ket_qua_tt = select_label(ket_qua_thi_thu, "Prize").ravel()

ket_qua_pvhoi6 = pd.read_csv('ket_qua_pvhoi6_utf8.csv')
feature_ket_qua_pvhoi6 = select_features(ket_qua_pvhoi6, ["A", "B", "C", "D", "E", "F", "Points"])
target_ket_qua_pvhoi6 = select_label(ket_qua_pvhoi6, "Prize").ravel()

ket_qua_hsgqg25 = pd.read_csv('ket_qua_hsgqg25_utf8.csv')
feature_ket_qua_hsgqg25 = select_features(ket_qua_hsgqg25, ["Score"])
target_ket_qua_hsgqg25 = select_label(ket_qua_hsgqg25, "Prize").ravel()

X_train_tt, X_test_tt, y_train_tt, y_test_tt = train_test_split(feature_ket_qua_tt, target_ket_qua_tt, test_size=0.2, random_state=42)
X_train_pv, X_test_pv, y_train_pv, y_test_pv = train_test_split(feature_ket_qua_pvhoi6, target_ket_qua_pvhoi6, test_size=0.2, random_state=42)  
X_train_hsgqg25, X_test_hsgqg25, y_train_hsgqg25, y_test_hsgqg25 = train_test_split(feature_ket_qua_hsgqg25, target_ket_qua_hsgqg25, test_size=0.2, random_state=42)  

# == Train models ==
@st.cache_resource
def load_models():
    model_tt = RandomForestClassifier(n_estimators=200, random_state=42)
    model_pv = RandomForestClassifier(n_estimators=200, random_state=42)
    model_hsgqg25 = RandomForestClassifier(n_estimators=200, random_state=42)

    model_tt.fit(X_train_tt, y_train_tt)
    model_pv.fit(X_train_pv, y_train_pv)
    model_hsgqg25.fit(X_train_hsgqg25, y_train_hsgqg25)

    return model_tt, model_pv, model_hsgqg25

model_tt, model_pv, model_hsgqg25 = load_models()

# == Streamlit app ==
st.title("🎯 Dự đoán giải thưởng")

A = st.number_input("Bai1", 0.0)
B = st.number_input("Bai2", 0.0)
C = st.number_input("Bai3", 0.0)
D = st.number_input("Bai4", 0.0)
E = st.number_input("Bai5", 0.0)
F = st.number_input("Bai6", 0.0)

def prize_to_text(prize):
    return {
        1: "Giải Nhất",
        2: "Giải Nhì",
        3: "Giải Ba",
        4: "Giải Khuyến khích",
        5: "Không đạt giải"
    }.get(prize, "Không xác định")

# if st.button("Dự đoán"):
#     points = A + B + C + D + E + F
#     X_new = np.array([[A, B, C, D, E, F, points]])

#     p1 = model_pv.predict(X_new)[0]
#     p2 = model_tt.predict(X_new)[0]
#     X_new_hsgqg25 = np.array([[points]])
#     p3 = model_hsgqg25.predict(X_new_hsgqg25)[0]

#     from collections import Counter

#     prize = Counter([p1, p2, p3]).most_common(1)[0][0]

#     st.success(f"Kết quả: {prize_to_text(prize)}")

if st.button("Dự đoán"):
    points = A + B + C + D + E + F
    X_new_pv = np.array([[A, B, C, D, E, F, points]])
    X_new_tt = X_new_pv.copy()
    X_new_hsg = np.array([[points]])

    # Helper: lấy dict {class: prob} từ model (nếu model hỗ trợ predict_proba)
    def probs_dict(model, Xnew):
        if not hasattr(model, "predict_proba"):
            return {}
        probs = model.predict_proba(Xnew)[0]
        classes = model.classes_
        # chuyển class về int nếu cần
        return {int(c): float(p) for c, p in zip(classes, probs)}

    try:
        p_pv = probs_dict(model_pv, X_new_pv)
        p_tt = probs_dict(model_tt, X_new_tt)
        p_hsg = probs_dict(model_hsgqg25, X_new_hsg)

        # combine (trung bình xác suất từ 3 model) cho các lớp 1..5
        classes_all = [1, 2, 3, 4, 5]
        combined = {}
        for k in classes_all:
            combined[k] = (p_pv.get(k, 0.0) + p_tt.get(k, 0.0) + p_hsg.get(k, 0.0)) / 3.0

        # predicted by max combined prob
        pred_by_prob = max(combined.items(), key=lambda x: x[1])[0]

        # predicted by majority vote (giữ logic hiện tại)
        from collections import Counter
        p1 = int(model_pv.predict(X_new_pv)[0])
        p2 = int(model_tt.predict(X_new_tt)[0])
        p3 = int(model_hsgqg25.predict(X_new_hsg)[0])
        pred_by_vote = Counter([p1, p2, p3]).most_common(1)[0][0]

        # Hiển thị kết quả
        st.success(f"Kết quả (majority vote): {prize_to_text(pred_by_vote)}")
        st.info(f"Kết quả (max probability): {prize_to_text(pred_by_prob)}")

        # Hiển thị bảng phần trăm
        df_prob = pd.DataFrame({
            "Prize": [prize_to_text(k) for k in classes_all],
            "Probability (%)": [round(combined[k]*100, 2) for k in classes_all]
        })
        st.table(df_prob)

    except Exception as e:
        st.error("Không thể tính xác suất: " + str(e))


# == End of code ==
# 3.5, 0, 2.1, 4.2, 4.2, 0.1
