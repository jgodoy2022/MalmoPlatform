"""
Script de Evaluación para modelos entrenados con Stable-Baselines3
Carga y evalúa los modelos PPO y DQN en el entorno de Malmo
"""
import argparse
import malmoenv
import numpy as np
from pathlib import Path
from lxml import etree
from threading import Thread, Barrier
import time
from stable_baselines3 import PPO, DQN
import json

# Importar el wrapper del script de entrenamiento
import sys
sys.path.append('.')
from train_chase_escape_opt import MalmoGymWrapper


def select_model_path(role_name, user_path=None, prefer_best=True):
    """
    Devuelve la ruta del modelo a usar.
    Orden de prioridad: ruta manual > *_best.zip > *_FINAL.zip.
    """
    candidates = []
    if user_path:
        candidates.append(Path(user_path))

    if prefer_best:
        candidates.append(Path(f"models/{role_name.lower()}_best.zip"))
        candidates.append(Path(f"models/{role_name}_best.zip"))

    candidates.append(Path(f"models/{role_name}_FINAL.zip"))
    candidates.append(Path(f"models/{role_name.lower()}_final.zip"))

    for path in candidates:
        if path.exists():
            return path

    readable = "\n    ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No se encontr\u00f3 ning\u00fan modelo para {role_name}. Se buscaron:\n    {readable}"
    )


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
        
        print(f"[{role_name}] ✓ Modelo cargado, iniciando evaluación...")
        
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
        print(f"[{role_name}] ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("=" * 70)
    print("  EVALUACIÓN DE MODELOS ENTRENADOS")
    print("=" * 70)

    parser = argparse.ArgumentParser(description="Evalúa modelos PPO/DQN entrenados en Malmo.")
    parser.add_argument(
        "--ppo-model",
        type=str,
        help="Ruta manual al zip del Perseguidor (por defecto usa *_best.zip o *_FINAL.zip)."
    )
    parser.add_argument(
        "--dqn-model",
        type=str,
        help="Ruta manual al zip del Escapista (por defecto usa *_best.zip o *_FINAL.zip)."
    )
    parser.add_argument(
        "--prefer-final",
        action="store_true",
        help="Usar *_FINAL.zip aunque exista un *_best.zip."
    )
    args = parser.parse_args()

    # Configuración
    PORT = 9000
    SERVER = '127.0.0.1'
    N_EPISODES = 10

    try:
        ppo_model_path = select_model_path(
            role_name="Perseguidor_PPO",
            user_path=args.ppo_model,
            prefer_best=not args.prefer_final
        )
        dqn_model_path = select_model_path(
            role_name="Escapista_DQN",
            user_path=args.dqn_model,
            prefer_best=not args.prefer_final
        )
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        exit(1)

    # Cargar XML
    xml_path = Path('missions/chase_escape.xml')
    if not xml_path.exists():
        print(f"\nERROR: No se encuentra {xml_path}")
        exit(1)
    
    xml = xml_path.read_text()
    mission = etree.fromstring(xml)
    number_of_agents = len(mission.findall('{http://ProjectMalmo.microsoft.com}AgentSection'))
    
    print(f"\nConfiguración:")
    print(f"  Episodios de evaluación: {N_EPISODES}")
    print(f"  Modelo PPO: {ppo_model_path}")
    print(f"  Modelo DQN: {dqn_model_path}")
    if args.prefer_final:
        print("  Preferencia de carga: *_FINAL.zip")
    elif args.ppo_model or args.dqn_model:
        print("  Preferencia de carga: rutas manuales cuando se indican; luego *_best.zip/_FINAL.zip")
    else:
        print("  Preferencia de carga: *_best.zip si existe, si no *_FINAL.zip")
    
    # Instrucciones
    print("\n" + "=" * 70)
    print("ANTES DE CONTINUAR:")
    print("  1. Abre DOS terminales")
    print(f"  2. Terminal 1: py -3.7 -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft({PORT})\"")
    print(f"  3. Terminal 2: py -3.7 -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft({PORT + 1})\"")
    print("  4. Espera a que AMBAS instancias estén listas")
    print("=" * 70)
    print("\nPresiona ENTER cuando estés listo...")
    input()
    
    print("\nIniciando evaluación...")
    time.sleep(2)
    
    # Resultados compartidos
    results_dict = {}
    
    # Barrier
    start_barrier = Barrier(number_of_agents)
    
    # Crear threads
    threads = [
        Thread(
            target=evaluate_agent,
            args=(
                0, xml, PORT, SERVER, SERVER,
                ppo_model_path, N_EPISODES,
                start_barrier, results_dict
            ),
            name="Perseguidor"
        ),
        Thread(
            target=evaluate_agent,
            args=(
                1, xml, PORT, SERVER, SERVER,
                dqn_model_path, N_EPISODES,
                start_barrier, results_dict
            ),
            name="Escapista"
        )
    ]
    
    [t.start() for t in threads]
    [t.join() for t in threads]
    
    # Mostrar resultados finales
    print("\n" + "=" * 70)
    print("  RESULTADOS DE LA EVALUACIÓN")
    print("=" * 70)
    
    for role in [0, 1]:
        if role in results_dict:
            res = results_dict[role]
            print(f"\n{res['role_name']}:")
            print(f"  Reward promedio: {res['mean_reward']:.2f} +/- {res['std_reward']:.2f}")
            print(f"  Mejor reward: {max(res['all_rewards']):.2f}")
            print(f"  Peor reward: {min(res['all_rewards']):.2f}")
            print(f"  Pasos promedio: {res['mean_length']:.1f}")
            print(f"  Victorias: {res['wins']}/{N_EPISODES} ({res['win_rate']*100:.1f}%)")
    
    # Guardar resultados en JSON
    results_file = f"evaluation_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResultados guardados en: {results_file}")
    print("\n" + "=" * 70)
    print("Evaluación completada")
    print("=" * 70)
