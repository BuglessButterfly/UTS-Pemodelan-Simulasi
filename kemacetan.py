import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# 1. LOAD DATASET
# ============================================================
df = pd.read_csv("Prediksi Lalu Lintas.csv")

# Gunakan kolom Total kendaraan sebagai indikator kemacetan
traffic = df['Total'].values

# Representasi waktu sebagai index (t = 0, 1, 2, ..., n)
t = np.arange(len(traffic))

# ============================================================
# 2. SISTEM DINAMIK: MODEL STOCK–FLOW
#    STOCK = jumlah kendaraan di jalan
#    FLOW = kendaraan masuk - kendaraan keluar
# ============================================================

# Kita definisikan model sistem dinamik:
# dV/dt = inflow(t) - outflow(t)
# Volume(t) = V0 * exp(r * t)    → r adalah net flow rate

def sd_model(t, V0, r):
    return V0 * np.exp(r * t)

# Estimasi parameter sistem dinamik
params, _ = curve_fit(sd_model, t, traffic, p0=[traffic[0], 0.0001])
V0_opt, r_opt = params

# Prediksi model sistem dinamik
t_pred = np.linspace(0, len(traffic), 500)
traffic_pred = sd_model(t_pred, V0_opt, r_opt)

# ============================================================
# 3. HITUNG INFLOW DAN OUTFLOW
# ============================================================
# Net Flow = dV/dt = r * V(t)
net_flow = r_opt * traffic

# Jika net_flow > 0 → kendaraan bertambah (potensi macet)
# Jika net_flow < 0 → kendaraan berkurang (lancarnya lalu lintas)

# ============================================================
# 4. PLOTTING SISTEM DINAMIK KEMACETAN
# ============================================================
plt.figure(figsize=(12, 7))

# Data asli
plt.scatter(t, traffic, s=12, label="Data Asli Total Kendaraan")

# Model sistem dinamik
plt.plot(
    t_pred,
    traffic_pred,
    linewidth=2,
    label=f"Model Sistem Dinamik (r = {r_opt:.6f})"
)

plt.title("Model Sistem Dinamik Kemacetan Lalu Lintas")
plt.xlabel("Waktu (Index)")
plt.ylabel("Jumlah Kendaraan (Stock)")
plt.grid(True)

# Garis awal
plt.axhline(
    traffic[0],
    color='gray',
    linestyle='--',
    label=f"Volume Awal ({traffic[0]})"
)

plt.legend()
plt.show()   # <- HANYA MENAMPILKAN, TIDAK MENYIMPAN

# ============================================================
# 5. ANALISIS ARUS KENDARAAN (FLOW)
# ============================================================
plt.figure(figsize=(12, 5))
plt.plot(t, net_flow, label="Net Flow (Inflow - Outflow)")
plt.axhline(0, color='black', linestyle='--')
plt.title("Analisis Flow Lalu Lintas (Net Flow)")
plt.xlabel("Waktu")
plt.ylabel("Net Flow Kendaraan")
plt.grid(True)
plt.legend()
plt.show()   # <- JUGA HANYA MENAMPILKAN, TIDAK MENYIMPAN