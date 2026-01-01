import pandas as pd
import streamlit as st
import numpy as np

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def select_features(df, feature_name):
    return df[feature_name].values

def select_target(df, label_name):
    return df[label_name].values

ket_qua_hsgqg25 = pd.read_csv('ket_qua_hsgqg25_utf8.csv')
feature_ket_qua_hsgqg25 = select_features(ket_qua_hsgqg25, ["Score"])
target_ket_qua_hsgqg25 = select_target(ket_qua_hsgqg25, "Prize").ravel()

X_train_hsgqg25, X_test_hsgqg25, y_train_hsgqg25, y_test_hsgqg25 = train_test_split(feature_ket_qua_hsgqg25, target_ket_qua_hsgqg25, test_size=0.2, random_state=42)  


@st.cache_resource
def get_fitted_stacking_model(X, y):
    base_model_hsgqg25 = [
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)),
        ('svc', SVC(kernel='rbf', C=1, gamma='scale', probability=True)),
    ]
    metal_model_hsgqg25 = KNeighborsClassifier(n_neighbors=5)
    stacking = StackingClassifier(estimators=base_model_hsgqg25, final_estimator=metal_model_hsgqg25)
    stacking.fit(X, y)
    return stacking

stacking_model_hsgqg25 = get_fitted_stacking_model(X_train_hsgqg25, y_train_hsgqg25)

st.title("👀 Your VOI's prize issss ...... ?")

A = st.number_input("Bài 1 (max_score = 7)", min_value=0.0, max_value=7.0, step=0.1)
B = st.number_input("Bài 2 (max_score = 7)", min_value=0.0, max_value=7.0, step=0.1)
C = st.number_input("Bài 3 (max_score = 6)", min_value=0.0, max_value=6.0, step=0.1)
D = st.number_input("Bài 4 (max_score = 7)", min_value=0.0, max_value=7.0, step=0.1)
E = st.number_input("Bài 5 (max_score = 7)", min_value=0.0, max_value=7.0, step=0.1)
F = st.number_input("Bài 6 (max_score = 6)", min_value=0.0, max_value=6.0, step=0.1)
max_total = 7+7+6+7+7+6
points = A+B+C+D+E+F

st.metric("Tổng điểm", f"{points:.2f} / {max_total}")

if "play_music" not in st.session_state:
    st.session_state.play_music = False

if st.button("Submit"):
    st.session_state.play_music = True
    points = A + B + C + D + E + F
    X_new_hsgqg25 = np.array([[points]])
    proba = stacking_model_hsgqg25.predict_proba(X_new_hsgqg25)[0]

    best_idx = np.argmax(proba)

    prize_names = [
        "Giải Nhất 😍",
        "Giải Nhì 😘",
        "Giải Ba 😅",
        "Giải Khuyến khích 🙄",
        "Tạch 💀"
    ]

    df_result = pd.DataFrame({
        "Giải": prize_names,
        "Xác suất (%)": (proba * 100).round(2)
    })

    df_result["⭐"] = ""
    df_result.loc[best_idx, "⭐"] = "⬅️ Dự đoán cao nhất"

    st.dataframe(df_result, use_container_width=True)

import base64

if st.session_state.play_music:
    with open("JoJo's meme music - Wason Exploding.mp3", "rb") as f:
        audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()

    st.markdown(
        f"""
        <audio autoplay loop>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True
    )



