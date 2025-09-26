import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_trainer import load_and_train

st.set_page_config(page_title="🔥 Calorie Burn Predictor by Narmadha", layout="centered")

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("https://assets-v2.lottiefiles.com/a/dc68e41e-1189-11ee-a704-a3ee683b17ee/ZYGSIW3f8a.gif");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    .main h1 {{
        font-family: 'Segoe UI', sans-serif;
        color: Black;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(255, 105, 180, 0.3);
    }}

    .dark-subheading {{
        font-family: 'Segoe UI', sans-serif;
        color: Red;
        font-size: 2.0rem;
        font-weight: 800;
        margin-bottom: 20px;
        text-align: center;
    }}

    .stButton>button {{
        background-color: Red;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }}
    .stButton>button:hover {{
        background-color: Green;
        color: white;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main">🔥 Calorie Burn Predictor</h1>', unsafe_allow_html=True)

results = load_and_train()

with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="dark-subheading">🎽 Enter Your Workout Details</div>', unsafe_allow_html=True)

    gender = st.selectbox("🧍 Gender", ["male", "female"])
    age = st.slider("🎂 Age", 15, 80, 30)
    height = st.slider("📏 Height (cm)", 140, 220, 170)
    weight = st.slider("⚖️ Weight (kg)", 40, 150, 70)
    duration = st.slider("⏱️ Workout Duration (min)", 10, 120, 45)
    heart_rate = st.slider("❤️ Heart Rate (bpm)", 60, 200, 100)
    body_temp = st.slider("🌡️ Body Temperature (°C)", 36.0, 42.0, 37.0)
    activity_type = st.selectbox("🏃‍♀️ Select Your Workout Type", ["Walking", "Running", "Cycling", "Swimming", "Yoga"])

    activity_factor = {
        "Walking": 1.0,
        "Running": 1.3,
        "Cycling": 1.2,
        "Swimming": 1.4,
        "Yoga": 0.8
    
    }

    if st.button("🎯 Predict Calories"):
        gender_val = 0 if gender == "male" else 1
        bmi = weight / ((height / 100) ** 2)

        input_data = pd.DataFrame([[gender_val, age, height, weight, duration, heart_rate, body_temp, bmi]],
                                  columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI'])

        st.markdown('<div class="dark-subheading">📊 Estimated Calories Burned</div>', unsafe_allow_html=True)

        for model, res in results.items():
            prediction = res['Model'].predict(input_data)[0]
            prediction *= activity_factor[activity_type]

            st.markdown(f"""
                <div style="
                    background-color: Black;
                    padding: 15px;
                    border-radius: 12px;
                    margin-top: 10px;
                    margin-bottom: 10px;
                    color: Red;
                    font-weight: bold;
                    font-size: 1.2rem;
                    text-align: center;
                    box-shadow: 0 0 15px rgba(0,0,0,0.3);">
                    💪 {model}: <span style="color: #ff69b4;">{prediction:.2f} kcal</span>
                </div>
            """, unsafe_allow_html=True)


            st.markdown("### 💬 Health Tip Based on Your Workout:")
            if prediction < 150:
                st.info("🚶‍♂️ Light burn — try increasing duration or choosing a higher-intensity activity.")
            elif 150 <= prediction <= 350:
                st.success("✅ Moderate burn — Great job! Stay consistent. 💪")
            else:
                st.balloons()
                st.success("🔥 High burn! You're smashing it! Stay hydrated and fuel your body. 🥗")

            st.markdown("### 📊 BMI Status:")
            if bmi < 18.5:
                st.warning("Underweight 😕")
            elif 18.5 <= bmi < 24.9:
                st.success("Normal 😊")
            elif 25 <= bmi < 29.9:
                st.warning("Overweight 😐")
            else:
                st.error("Obese 😟")

            durations = list(range(10, 130, 10))
            calories = []
            for d in durations:
                temp_input = input_data.copy()
                temp_input["Duration"] = d
                temp_pred = res['Model'].predict(temp_input)[0] * activity_factor[activity_type]
                calories.append(temp_pred)

            fig, ax = plt.subplots()
            ax.plot(durations, calories, color='Blue')
            ax.set_title("Calories Burned vs Duration")
            ax.set_xlabel("Duration (min)")
            ax.set_ylabel("Calories Burned (kcal)")
            st.pyplot(fig)

            if st.button("💾 Save This Result", key=f"save_result_{model}"):
                history = pd.DataFrame({
                    "Gender": [gender],
                    "Age": [age],
                    "Activity": [activity_type],
                    "Calories_Burned": [prediction]
                })
                history.to_csv("calorie_prediction_history.csv", index=False)
                st.success("✅ Prediction saved to calorie_prediction_history.csv")

    st.markdown("</div>", unsafe_allow_html=True)
