import pandas as pd
import matplotlib.pyplot as plt

# --- Cargar CSV ---
df = pd.read_csv("stats/Perseguidor_PPO_episodes.csv")

# --- Crear eje X como contador de filas ---
df["episodio_global"] = range(1, len(df) + 1)

# --- Victorias acumuladas ---
df["victorias_perseguidor"] = df["captured"].cumsum()
df["victorias_escapista"] = (1 - df["captured"]).cumsum()
def evaluate_agent(role, xml, port, server, server2, model_path, n_episodes, start_barrier, results_dict):
    """
    Evalúa un agente entrenado
    """
    role_name = "Perseguidor_PPO" if role == 0 else "Escapista_DQN"
    
    print(f"\n[{role_name}] Cargando modelo desde {model_path}...")
    
    # Esperar sincronización
    start_barrier.wait()
    time.sleep(role * 2)
    
    try:
        # Crear entorno
        env = MalmoGymWrapper(
            xml=xml,
            port=port,
            server=server,
            server2=server2,
            port2=port + 1,
            role=role,
            exp_uid='sb3_evaluation'
        )
        
        # Cargar modelo
        if role == 0:
            model = PPO.load(model_path, env=env)
        else:
            model = DQN.load(model_path, env=env)
        
        print(f"[{role_name}] Modelo cargado, iniciando evaluación...")
        
        # Estadísticas
        episode_rewards = []
        episode_lengths = []
        wins = 0
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            print(f"[{role_name}] Episodio {episode + 1}/{n_episodes}")
            
            while not done and steps < 500:
                # Predecir acción (sin exploración)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                
                time.sleep(0.02)  # Más lento para visualización
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            # Detectar victoria
            if role == 0 and episode_reward > 50:  # Perseguidor capturó
                wins += 1
            elif role == 1 and episode_reward > 50:  # Escapista sobrevivió
                wins += 1
            
            print(f"  Reward: {episode_reward:.2f}, Steps: {steps}")
        
        # Guardar resultados
        results_dict[role] = {
            'role_name': role_name,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'wins': wins,
            'win_rate': wins / n_episodes,
            'all_rewards': [float(r) for r in episode_rewards],
            'all_lengths': [int(l) for l in episode_lengths]
        }
        
        print(f"\n[{role_name}] Evaluación completada:")
        print(f"  Reward promedio: {results_dict[role]['mean_reward']:.2f} ± {results_dict[role]['std_reward']:.2f}")
        print(f"  Pasos promedio: {results_dict[role]['mean_length']:.1f}")
        print(f"  Tasa de victoria: {results_dict[role]['win_rate']*100:.1f}%")
        
        env.close()
    
    except Exception as e:
        print(f"[{role_name}] ERROR: {e}")
        import traceback
        traceback.print_exc()

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
