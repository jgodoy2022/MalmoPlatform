import pandas as pd
import matplotlib.pyplot as plt

# --- Cargar CSV ---
df = pd.read_csv("stats/Perseguidor_PPO_episodes.csv")

# --- Crear eje X como contador de filas ---
df["episodio_global"] = range(1, len(df) + 1)

# --- Victorias acumuladas ---
df["victorias_perseguidor"] = df["captured"].cumsum()
df["victorias_escapista"] = (1 - df["captured"]).cumsum()

# --- Graficar ---
plt.figure(figsize=(12, 6))

plt.plot(df["episodio_global"], df["victorias_perseguidor"], label="Perseguidor (victorias)", linewidth=2)
plt.plot(df["episodio_global"], df["victorias_escapista"], label="Escapista (victorias)", linewidth=2)

plt.title("Victorias acumuladas (episodios continuos)", fontsize=16)
plt.xlabel("Número total de episodios", fontsize=14)
plt.ylabel("Victorias acumuladas", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
