import os
import io
import base64

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "Prediksi Lalu Lintas.csv"

app = Flask(__name__)

# ============================================================
# LOADING DATASET (SAFE)
# ============================================================
def load_dataset(path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        engine="python",
        skip_blank_lines=False,
        keep_default_na=False,
        na_filter=False,
    )

    # bersihkan kolom
    df.columns = [c.strip() for c in df.columns]

    expected = [
        "Time",
        "Date",
        "Day of the week",
        "CarCount",
        "BikeCount",
        "BusCount",
        "TruckCount",
        "Total",
        "Traffic Situation",
    ]

    for c in expected:
        if c not in df.columns:
            df[c] = ""

    return df.reset_index(drop=True)


df = load_dataset()

# ============================================================
# TIME PARSER
# ============================================================
def time_to_minutes(tstr: str) -> int:
    try:
        s = str(tstr).strip()
        if not s:
            return 0

        # Jika ada AM/PM (contoh: "12:00:00 AM" atau "12:00 AM")
        parts = s.split()
        if len(parts) == 2:
            timepart, ampm = parts
            hh_mm = (timepart.split(":") + ["0", "0"])[:3]
            hh = int(hh_mm[0]) if hh_mm[0].isdigit() else 0
            mm = int(hh_mm[1]) if hh_mm[1].isdigit() else 0

            if ampm.upper() == "PM" and hh != 12:
                hh += 12
            if ampm.upper() == "AM" and hh == 12:
                hh = 0

            return hh * 60 + mm

        # Jika format 24 jam "HH:MM" atau "HH"
        hh_mm = (s.split(":") + ["0"])[:2]
        hh = int(hh_mm[0]) if hh_mm[0].isdigit() else 0
        mm = int(hh_mm[1]) if hh_mm[1].isdigit() else 0
        return hh * 60 + mm

    except Exception:
        return 0


# ============================================================
# FEATURE PREPARATION / CLASSIFIER TANPA SCIKIT-LEARN
# ============================================================
NUM_COLS = ["CarCount", "BikeCount", "BusCount", "TruckCount", "Total"]


def prepare_numeric(df_input: pd.DataFrame) -> pd.DataFrame:
    df_local = df_input.copy()

    # pastikan kolom tersedia
    for c in ["Time", "Date", "Day of the week"] + NUM_COLS:
        if c not in df_local.columns:
            df_local[c] = ""

    # Time -> minutes
    df_local["Time_min"] = df_local["Time"].astype(str).apply(time_to_minutes)

    # Date
    df_local["Date"] = df_local["Date"].astype(str).str.replace(" ", "")
    df_local["Date"] = df_local["Date"].replace("", "0")
    df_local["Date"] = pd.to_numeric(df_local["Date"], errors="coerce").fillna(0).astype(
        int
    )

    # Numeric columns
    for col in NUM_COLS:
        df_local[col] = df_local[col].astype(str).str.replace(" ", "")
        df_local[col] = df_local[col].replace("", "0")
        df_local[col] = pd.to_numeric(df_local[col], errors="coerce").fillna(0).astype(
            float
        )

    return df_local


# "Fitur" yang dipakai classifier sederhana (tanpa scikit-learn)
FEATURE_COLS = ["Time_min", "CarCount", "BikeCount", "BusCount", "TruckCount", "Total"]

# hitung centroid per kelas (Traffic Situation) di awal
df_num = prepare_numeric(df)
unique_classes = sorted(df_num["Traffic Situation"].astype(str).unique())

CLASS_CENTROIDS = {}
for cls in unique_classes:
    sub = df_num[df_num["Traffic Situation"].astype(str) == cls]
    if len(sub) == 0:
        continue
    centroid = sub[FEATURE_COLS].to_numpy(dtype=float).mean(axis=0)
    CLASS_CENTROIDS[cls] = centroid


def classifier_predict_proba(row_dict):
    """
    Klasifier sederhana:
    - ambil fitur numeric
    - hitung jarak ke centroid tiap kelas
    - konversi jarak menjadi "probabilitas" via softmax(-dist)
    """
    single_df = pd.DataFrame([row_dict])
    single_num = prepare_numeric(single_df)
    x = single_num[FEATURE_COLS].to_numpy(dtype=float)[0]

    labels = list(CLASS_CENTROIDS.keys())
    if not labels:
        # fallback jika centroids kosong
        return ["unknown"], np.array([1.0])

    dists = []
    for cls in labels:
        c = CLASS_CENTROIDS[cls]
        d = np.linalg.norm(x - c)
        dists.append(d)

    dists = np.array(dists, dtype=float)

    # ubah jarak menjadi skor (semakin dekat semakin besar)
    # skor = exp(-dist)
    scores = np.exp(-dists + dists.min())
    probs = scores / scores.sum()

    return labels, probs


# ============================================================
# SISTEM DINAMIK (TANPA SCIPY)
# ============================================================
# convert traffic to numeric (fallback ke 0 jika gagal)
traffic_series = (
    df["Total"].astype(str).str.replace(" ", "", regex=False)
    if "Total" in df.columns
    else pd.Series([], dtype=str)
)
traffic = (
    pd.to_numeric(traffic_series, errors="coerce").fillna(0).astype(float).values
    if len(traffic_series) > 0
    else np.array([], dtype=float)
)
t = np.arange(len(traffic))


def sd_model(t_arr, V0, r):
    return V0 * np.exp(r * t_arr)


# estimasi parameter tanpa curve_fit: pakai regresi linier di log(traffic)
if len(traffic) > 1:
    safe_traffic = np.maximum(traffic, 1)
    coef = np.polyfit(t, np.log(safe_traffic), 1)
    r_opt = float(coef[0])
    V0_opt = float(np.exp(coef[1]))
elif len(traffic) == 1:
    V0_opt = float(traffic[0])
    r_opt = 0.0
else:
    V0_opt = 0.0
    r_opt = 0.0

t_pred = np.linspace(0, max(0, len(traffic) - 1), 400)
traffic_pred = sd_model(t_pred, V0_opt, r_opt) if len(traffic) else np.array([])

net_flow = r_opt * traffic if len(traffic) else np.array([])


# ============================================================
# UTIL: FIGURE -> BASE64
# ============================================================
def fig_to_base64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    plt.close()
    return base64.b64encode(img_bytes).decode("ascii")


# ============================================================
# PLOT GENERATORS (RETURN BASE64)
# ============================================================
def generate_main_plot():
    plt.figure(figsize=(10, 6))
    if len(t) and len(traffic):
        plt.scatter(t, traffic, s=15, label="Data")
        plt.plot(t_pred, traffic_pred, lw=2, label="Model")
        plt.legend()
    plt.title("Dynamic Traffic Model")
    plt.xlabel("Index")
    plt.ylabel("Total Vehicles")
    plt.tight_layout()
    return fig_to_base64()


def plot_volume():
    plt.figure(figsize=(7, 4))
    if len(traffic):
        plt.plot(traffic, linewidth=2)
    plt.title("Traffic Volume Over Time")
    plt.xlabel("Index")
    plt.ylabel("Total Vehicles")
    plt.tight_layout()
    return fig_to_base64()


def plot_composition():
    numeric_cols = ["CarCount", "BikeCount", "BusCount", "TruckCount"]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0
    avg_vals = df[numeric_cols].astype(float).mean()
    plt.figure(figsize=(7, 4))
    plt.bar(avg_vals.index, avg_vals.values)
    plt.title("Average Vehicle Composition")
    plt.ylabel("Average Count")
    plt.tight_layout()
    return fig_to_base64()


def plot_netflow():
    plt.figure(figsize=(7, 4))
    if len(net_flow):
        plt.plot(net_flow, linewidth=2)
    plt.title("Net Flow Dynamics")
    plt.xlabel("Index")
    plt.ylabel("Net Flow")
    plt.tight_layout()
    return fig_to_base64()


def generate_class_chart_for_label(label_str, base_row):
    """
    Membuat grafik khusus untuk 1 kelas (Traffic Situation) dalam bentuk base64.
    Dipakai untuk menampilkan pola pada input & centroid.
    """
    plt.figure(figsize=(7, 4))

    # series: [Car, Bike, Bus, Truck, Total]
    x_labels = ["Car", "Bike", "Bus", "Truck", "Total"]
    vals = [
        base_row.get("CarCount", 0),
        base_row.get("BikeCount", 0),
        base_row.get("BusCount", 0),
        base_row.get("TruckCount", 0),
        base_row.get("Total", 0),
    ]
    vals = [float(str(v).strip() or 0) for v in vals]

    plt.bar(x_labels, vals)
    plt.title(f"Traffic Composition — {label_str}")
    plt.ylabel("Count")
    plt.tight_layout()
    return fig_to_base64()


def generate_probability_bar(labels, probs):
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probs)
    plt.ylim(0, 1)
    plt.title("Prediction Probabilities (approx.)")
    plt.tight_layout()
    return fig_to_base64()


# generate base64 charts at startup (dipakai berulang)
MAIN_PLOT_B64 = generate_main_plot()
CHART_VOLUME_B64 = plot_volume()
CHART_COMP_B64 = plot_composition()
CHART_NETFLOW_B64 = plot_netflow()


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def home():
    return render_template(
        "index.html",
        V0=round(float(V0_opt), 4) if V0_opt is not None else 0,
        r_value=f"{r_opt:.6f}" if r_opt is not None else "0.0",
        total_data=len(df),
        min_flow=float(net_flow.min()) if len(net_flow) else 0,
        max_flow=float(net_flow.max()) if len(net_flow) else 0,
        dataset_options=list(df.index),
        prediction_text=None,
        probability_text=None,
        color=None,
        bar_chart_data=None,
        class_chart_data=None,
        main_plot_data=MAIN_PLOT_B64,
        chart_volume_data=CHART_VOLUME_B64,
        chart_comp_data=CHART_COMP_B64,
        chart_net_data=CHART_NETFLOW_B64,
    )


@app.route("/get_dataset/<int:row_id>")
def get_dataset(row_id: int):
    if row_id < 0 or row_id >= len(df):
        return jsonify({"error": "Index not found"}), 404
    row = df.iloc[row_id]
    out = {
        "Time": row.get("Time", ""),
        "Date": row.get("Date", ""),
        "Day of the week": row.get("Day of the week", ""),
        "CarCount": row.get("CarCount", ""),
        "BikeCount": row.get("BikeCount", ""),
        "BusCount": row.get("BusCount", ""),
        "TruckCount": row.get("TruckCount", ""),
        "Total": row.get("Total", ""),
    }
    return jsonify(out)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    single = {
        "Time": form.get("Time", ""),
        "Date": form.get("Date", "0"),
        "Day of the week": form.get("Day of the week", ""),
        "CarCount": form.get("CarCount", "0"),
        "BikeCount": form.get("BikeCount", "0"),
        "BusCount": form.get("BusCount", "0"),
        "TruckCount": form.get("TruckCount", "0"),
        "Total": form.get("Total", "0"),
    }

    # prediksi dengan classifier sederhana
    labels, probs = classifier_predict_proba(single)
    idx = int(np.argmax(probs))
    predicted_label = labels[idx]
    top_prob = float(probs[idx])

    prob_text = f"Probabilitas (approx.): {top_prob:.3f} (Kelas {predicted_label})"

    # probability bar chart (base64)
    bar_b64 = generate_probability_bar(labels, probs)

    # class-specific chart (base64)
    class_b64 = generate_class_chart_for_label(predicted_label, single)

    # kategorisasi berdasarkan Total (kuartil dari traffic historis)
    try:
        total_input = int(str(single["Total"]).strip())
    except Exception:
        total_input = 0

    if len(traffic):
        p25, p50, p75 = np.percentile(traffic, [25, 50, 75])
    else:
        p25 = p50 = p75 = 0

    if total_input >= p75 and p75 > 0:
        cat, color = "Kemacetan Parah", "danger"
    elif total_input >= p50 and p50 > 0:
        cat, color = "Padat", "warning"
    elif total_input >= p25 and p25 > 0:
        cat, color = "Sedang", "info"
    else:
        cat, color = "Lancar", "success"

    prediction_text = f"{cat} — Prediksi model: {predicted_label} (Total={total_input})"

    return render_template(
        "index.html",
        V0=round(float(V0_opt), 4) if V0_opt is not None else 0,
        r_value=f"{r_opt:.6f}" if r_opt is not None else "0.0",
        total_data=len(df),
        min_flow=float(net_flow.min()) if len(net_flow) else 0,
        max_flow=float(net_flow.max()) if len(net_flow) else 0,
        dataset_options=list(df.index),
        prediction_text=prediction_text,
        probability_text=prob_text,
        color=color,
        bar_chart_data=bar_b64,
        class_chart_data=class_b64,
        main_plot_data=MAIN_PLOT_B64,
        chart_volume_data=CHART_VOLUME_B64,
        chart_comp_data=CHART_COMP_B64,
        chart_net_data=CHART_NETFLOW_B64,
    )


# ============================================================
# RUN SERVER (untuk lokal)
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
