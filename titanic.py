# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Titanic Explorer", layout="wide")

# --- Helpers ---
@st.cache_data
def load_data(path="titanic_data.csv"):
    df = pd.read_csv(path)
    return df

@st.cache_resource
def load_model(path="model.joblib"):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error("Model not found. Run train_model.py to create model.joblib.")
        return None

# Load
df = load_data()
model = load_model()

# Title
st.title("Titanic Explorer — Interactive Dashboard & Survival Predictor")

# Layout: left controls, right display
with st.sidebar:
    st.header("Filters")
    # Basic filters
    sexes = ["All"] + sorted(df["Sex"].dropna().unique().tolist())
    selected_sex = st.selectbox("Sex", sexes, index=0)
    classes = ["All"] + sorted(df["Pclass"].dropna().unique().tolist())
    selected_class = st.selectbox("Passenger Class (Pclass)", classes, index=0)
    boarded = ["All"] + sorted(df["Embarked"].dropna().unique().tolist())
    selected_embarked = st.selectbox("Embarked", boarded, index=0)
    age_range = st.slider("Age range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
    fare_max = int(df["Fare"].max())
    fare_range = st.slider("Fare range", 0, fare_max, (0, fare_max))

    st.markdown("---")
    st.write("Download filtered data:")
    # filtered download button created below after filtering

# Apply filters
filtered = df.copy()
if selected_sex != "All":
    filtered = filtered[filtered["Sex"] == selected_sex]
if selected_class != "All":
    filtered = filtered[filtered["Pclass"] == int(selected_class)]
if selected_embarked != "All":
    filtered = filtered[filtered["Embarked"] == selected_embarked]

filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]
filtered = filtered[(filtered["Fare"] >= fare_range[0]) & (filtered["Fare"] <= fare_range[1])]

# Main columns
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Dataset (filtered)")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

    # provide CSV download
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered CSV", csv, "titanic_filtered.csv", "text/csv")

with col2:
    st.subheader("Summary Statistics")
    total = len(filtered)
    survived = int(filtered["Survived"].sum())
    survival_rate = (survived / total * 100) if total > 0 else 0
    st.metric("Rows (current filters)", total)
    st.metric("Number survived (current filters)", survived)
    st.metric("Survival rate (current filters)", f"{survival_rate:.2f}%")

    st.markdown("**Overall dataset stats**")
    st.write(filtered[["Age","Fare","SibSp","Parch"]].describe().T)

# Charts
st.markdown("---")
st.subheader("Charts")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Survival count by Sex**")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered, x="Sex", hue="Survived", ax=ax1)
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Sex")
    ax1.legend(title="Survived", labels=["No","Yes"])
    st.pyplot(fig1)

with c2:
    st.markdown("**Survival rate by Pclass**")
    # compute survival rate per class
    class_rate = filtered.groupby("Pclass")["Survived"].mean().reset_index()
    fig2, ax2 = plt.subplots()
    sns.barplot(data=class_rate, x="Pclass", y="Survived", ax=ax2)
    ax2.set_ylabel("Survival Rate (0-1)")
    ax2.set_xlabel("Pclass")
    st.pyplot(fig2)

# Bonus: simple ML predictor UI
st.markdown("---")
st.subheader("Survival Predictor (Logistic Regression)")

if model is None:
    st.info("Model not loaded. Run `python train_model.py` to create model.joblib then refresh.")
else:
    with st.form("predict_form"):
        st.write("Enter passenger details:")
        pclass = st.selectbox("Pclass", sorted(df["Pclass"].unique()), index=0)
        sex = st.selectbox("Sex", sorted(df["Sex"].unique()), index=0)
        age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].median()))
        sibsp = st.number_input("SibSp (siblings/spouses)", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parch (parents/children)", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare", min_value=0.0, max_value=float(df["Fare"].max()), value=float(df["Fare"].median()))
        embarked = st.selectbox("Embarked", sorted(df["Embarked"].dropna().unique()))
        submitted = st.form_submit_button("Predict Survival")

    if submitted:
        sample = pd.DataFrame([{
            "Pclass": int(pclass),
            "Sex": sex,
            "Age": int(age),
            "SibSp": int(sibsp),
            "Parch": int(parch),
            "Fare": float(fare),
            "Embarked": embarked
        }])
        # Model returns 0/1 and we also show probability
        pred = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0][1]  # probability of survival
        st.write("### Prediction result")
        st.write("Predicted class:", "✅ Survived" if pred==1 else "❌ Not survived")
        st.write(f"Predicted survival probability: **{prob*100:.2f}%**")

