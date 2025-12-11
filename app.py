import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ========================================================
# LOAD MODEL DATA
# ========================================================
uploaded_file = "data.xlsx"

sheets = {
    "Catalyst dose": "Catalyst dose",
    "PMS dose": "PMS dose",
    "pH study": "pH study",
    "Power study": "Power study",
    "TET concentration": "TET concentration"
}

k_tables = {}
xls = pd.ExcelFile(uploaded_file)

for label, sheet in sheets.items():
    df = pd.read_excel(xls, sheet_name=sheet)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["k_obs"] = pd.to_numeric(df["k_obs"], errors="coerce")
    df = df.dropna(subset=["value", "k_obs"])
    k_tables[label] = df

# ========================================================
# TRAIN MODELS
# ========================================================
labels_units = {
    "Catalyst dose": ("Catalyst dose", "mg/L"),
    "PMS dose": ("PMS concentration", "mM"),
    "pH study": ("pH level", ""),
    "Power study": ("Microwave power", "W"),
    "TET concentration": ("Initial TET", "mg/L")
}

reg_models = {}

for key, df in k_tables.items():
    X = df[["value"]].values
    y = df["k_obs"].values
    model = LinearRegression()
    model.fit(X, y)
    reg_models[key] = {
        "model": model,
        "name": labels_units[key][0],
        "unit": labels_units[key][1]
    }

# ========================================================
# ALLOWED PARAMETER RANGES
# ========================================================
param_ranges = {
    "Catalyst dose": (5, 30),
    "PMS dose": (20, 120),
    "pH study": (1, 14),
    "Power study": (200, 700),
    "TET concentration": (1, 20)
}

# ========================================================
# STREAMLIT UI
# ========================================================
st.title("⭐ Interactive TET Degradation Model")
st.write("Predict **removal efficiency vs time** by changing any experimental parameter.")

# Parameter selection
parameter = st.selectbox("Select Parameter", list(reg_models.keys()))

p_min, p_max = param_ranges[parameter]

# Slider
value = st.slider("Adjust Value", min_value=float(p_min), max_value=float(p_max), step=0.5)

# Prediction
info = reg_models[parameter]
model = info["model"]
name, unit = info["name"], info["unit"]

k = model.predict(np.array([[value]]))[0]

t = np.linspace(0, 10, 200)
removal = (1 - np.exp(-k * t)) * 100

# ========================================================
# PLOT
# ========================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, removal, linewidth=3)
label_text = f"{name} = {value} {unit}" if unit else f"{name} = {value}"
ax.set_title(f"TET Removal vs Time ({label_text})")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Removal (%)")
ax.grid(alpha=0.3)
st.pyplot(fig)

# ========================================================
# PRINT VALUES
# ========================================================
st.subheader("🔍 Predicted Values")
st.write(f"**Predicted k_obs:** {k:.5f}")
st.write(f"**Removal at 1 min:** {((1 - np.exp(-k*1))*100):.2f}%")
st.write(f"**Removal at 3 min:** {((1 - np.exp(-k*3))*100):.2f}%")
st.write(f"**Removal at 5 min:** {((1 - np.exp(-k*5))*100):.2f}%")
st.write(f"**Removal at 6 min:** {((1 - np.exp(-k*6))*100):.2f}%")
