import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import io
import base64

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "Prediksi Lalu Lintas.csv"

app = Flask(__name__)

# ============================================================
# LOADING DATASET (SAFE)
# ============================================================
def load_dataset(path=CSV_PATH):
    df = pd.read_csv(
        path,
        dtype=str,
        engine="python",
        skip_blank_lines=False,
        keep_default_na=False,
        na_filter=False
    )

    # bersihkan kolom
    df.columns = [c.strip() for c in df.columns]

    expected = [
        "Time", "Date", "Day of the week",
        "CarCount", "BikeCount", "BusCount", "TruckCount",
        "Total", "Traffic Situation"
    ]

    for c in expected:
        if c not in df.columns:
            df[c] = ""

    return df.reset_index(drop=True)

df = load_dataset()

# ============================================================
# TIME PARSER
# ============================================================
def time_to_minutes(tstr):
    try:
        s = str(tstr).strip()
        if not s:
            return 0

        # Jika ada AM/PM (contoh: "12:00:00 AM" atau "12:00 AM")
        parts = s.split()
        if len(parts) == 2:
            timepart, ampm = parts
            hh_mm = (timepart.split(":") + ['0', '0'])[:3]
            hh = int(hh_mm[0]) if hh_mm[0].isdigit() else 0
            mm = int(hh_mm[1]) if hh_mm[1].isdigit() else 0

            if ampm.upper() == "PM" and hh != 12:
                hh += 12
            if ampm.upper() == "AM" and hh == 12:
                hh = 0

            return hh * 60 + mm

        # Jika format 24 jam "HH:MM" atau "HH"
        hh_mm = (s.split(":") + ['0'])[:2]
        hh = int(hh_mm[0]) if hh_mm[0].isdigit() else 0
        mm = int(hh_mm[1]) if hh_mm[1].isdigit() else 0
        return hh * 60 + mm

    except Exception:
        return 0

# ============================================================
# FEATURE PREPARATION (memastikan encoder tersimpan)
# ============================================================
def prepare_features(df_input, fit_encoders=False):
    """
    Mengembalikan X (numpy array) dan encoder yang dipakai (OneHotEncoder untuk hari).
    Jika fit_encoders=True maka day_encoder global akan di-set.
    """
    df_local = df_input.copy()

    # pastikan kolom tersedia
    expected = [
        "Time", "Date", "Day of the week",
        "CarCount", "BikeCount", "BusCount", "TruckCount", "Total"
    ]
    for c in expected:
        if c not in df_local.columns:
            df_local[c] = ""

    # Time -> minutes
    df_local["Time_min"] = df_local["Time"].astype(str).apply(time_to_minutes)

    # Date: bersihkan spasi, ubah "" -> "0", lalu ke numeric
    df_local["Date"] = df_local["Date"].astype(str).str.replace(" ", "")
    df_local["Date"] = df_local["Date"].replace("", "0")
    df_local["Date"] = pd.to_numeric(df_local["Date"], errors="coerce").fillna(0).astype(int)

    # Numeric columns: hilangkan spasi, ubah ""->"0", numeric
    for col in ["CarCount", "BikeCount", "BusCount", "TruckCount", "Total"]:
        df_local[col] = df_local[col].astype(str).str.replace(" ", "")
        df_local[col] = df_local[col].replace("", "0")
        df_local[col] = pd.to_numeric(df_local[col], errors="coerce").fillna(0).astype(int)

    # Day of the week -> one-hot
    days = df_local["Day of the week"].astype(str).replace("", "Unknown").to_numpy().reshape(-1, 1)

    global day_encoder

    # create OneHotEncoder in a way that's compatible across sklearn versions
    def make_encoder():
        try:
            return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            return OneHotEncoder(sparse=False, handle_unknown="ignore")

    if fit_encoders or "day_encoder" not in globals():
        enc = make_encoder()
        day_encoded = enc.fit_transform(days)
        day_encoder = enc  # simpan encoder global
    else:
        enc = day_encoder
        try:
            day_encoded = enc.transform(days)
        except Exception:
            enc = make_encoder()
            day_encoded = enc.fit_transform(days)
            day_encoder = enc

    # numeric features
    X_num = df_local[["Time_min", "Date", "CarCount", "BikeCount", "BusCount", "TruckCount", "Total"]].to_numpy()

    # gabungkan numeric + encoded day
    X = np.hstack([X_num, day_encoded])

    return X, enc

# ============================================================
# TRAIN MODEL
# ============================================================
label_col = "Traffic Situation"
labels = df[label_col].astype(str).values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# prepare features and encoder (fit on full df)
X_all, day_encoder = prepare_features(df, fit_encoders=True)

# safety trim (jika ada mismatch lengths)
n = min(len(df), len(X_all), len(y))
df = df.iloc[:n].reset_index(drop=True)
X_all = X_all[:n]
y = y[:n]

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_all, y)

# ============================================================
# SISTEM DINAMIK
# ============================================================
# convert traffic to numeric (fallback ke 0 jika gagal)
traffic = pd.to_numeric(df["Total"].astype(str).str.replace(" ", ""), errors="coerce").fillna(0).astype(float).values
t = np.arange(len(traffic))

def sd_model(t, V0, r):
    return V0 * np.exp(r * t)

try:
    params, _ = curve_fit(sd_model, t, traffic, p0=[traffic[0] if len(traffic) else 0, 0.0001], maxfev=20000)
    V0_opt, r_opt = float(params[0]), float(params[1])
except Exception:
    # fallback: linear fit on log
    safe_traffic = np.maximum(traffic, 1)
    coef = np.polyfit(t, np.log(safe_traffic), 1) if len(safe_traffic) > 1 else (0.0, np.log(safe_traffic[0]) if len(safe_traffic) else 0)
    r_opt = float(coef[0])
    V0_opt = float(np.exp(coef[1]))

t_pred = np.linspace(0, max(0, len(traffic) - 1), 400)
traffic_pred = sd_model(t_pred, V0_opt, r_opt) if len(traffic) else np.array([])

net_flow = r_opt * traffic if len(traffic) else np.array([])

# ============================================================
# UTIL: FIGURE -> BASE64 (BUKAN FILE STATIC)
# ============================================================
def fig_to_base64():
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
    numeric_cols = ["CarCount","BikeCount","BusCount","TruckCount"]
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

def generate_class_chart_for_label(label_str):
    """
    Membuat grafik khusus untuk 1 kelas (Traffic Situation) dalam bentuk base64.
    """
    mask = df[label_col].astype(str) == str(label_str)
    subset = df[mask]

    plt.figure(figsize=(7, 4))
    if len(subset):
        vals = pd.to_numeric(subset["Total"].astype(str).str.replace(" ", ""), errors="coerce").fillna(0).astype(float).values
        plt.plot(vals, linewidth=2)
        plt.title(f"Traffic Pattern — {label_str}")
        plt.xlabel("Index (subset)")
        plt.ylabel("Total Vehicles")
    else:
        plt.text(0.5, 0.5, f"No data for {label_str}", ha="center", va="center", fontsize=12)
        plt.title(f"Traffic Pattern — {label_str}")
        plt.axis('off')

    plt.tight_layout()
    return fig_to_base64()

def generate_probability_bar(probs):
    labels = list(label_encoder.classes_)
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probs)
    plt.ylim(0, 1)
    plt.title("Prediction Probabilities")
    plt.tight_layout()
    return fig_to_base64()

# generate base64 charts at startup (dipakai berulang)
MAIN_PLOT_B64 = generate_main_plot()
CHART_VOLUME_B64 = plot_volume()
CHART_COMP_B64 = plot_composition()
CHART_NETFLOW_B64 = plot_netflow()

# ============================================================
# HELPERS
# ============================================================
def row_to_dict(row: pd.Series):
    d = {}
    for k, v in row.items():
        if pd.isna(v):
            d[k] = ""
        else:
            try:
                d[k] = v.item() if hasattr(v, "item") else v
            except Exception:
                d[k] = str(v)
    return d

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
def get_dataset(row_id):
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
        "Total": row.get("Total", "")
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
        "Total": form.get("Total", "0")
    }

    df_single = pd.DataFrame([single])
    X_single, _ = prepare_features(df_single, fit_encoders=False)

    # safety: kalau shape mismatch, pad dengan zeros (jarang terjadi)
    if X_single.shape[1] != X_all.shape[1]:
        # coba rebuild day encoding dengan fit_encoders=True pada seluruh df, lalu transform single lagi
        prepare_features(df, fit_encoders=True)
        X_single, _ = prepare_features(df_single, fit_encoders=False)

        # jika masih mismatch, pad/crop
        if X_single.shape[1] < X_all.shape[1]:
            pad = np.zeros((X_single.shape[0], X_all.shape[1] - X_single.shape[1]))
            X_single = np.hstack([X_single, pad])
        elif X_single.shape[1] > X_all.shape[1]:
            X_single = X_single[:, :X_all.shape[1]]

    pred_probs = clf.predict_proba(X_single)[0]
    idx = int(np.argmax(pred_probs))
    label = label_encoder.inverse_transform([idx])[0]  # original label string

    prob_text = f"Probabilitas: {pred_probs[idx]:.3f} (Kelas {label})"

    # generate probability bar chart (base64)
    bar_b64 = generate_probability_bar(pred_probs)

    # class-specific chart (base64)
    class_b64 = generate_class_chart_for_label(label)

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

    prediction_text = f"{cat} — Prediksi model: {label} (Total={total_input})"

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
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)