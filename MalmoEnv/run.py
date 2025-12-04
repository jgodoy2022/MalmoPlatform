import malmoenv
import argparse
from pathlib import Path
import numpy as np
import random
import time
import os
from collections import deque
import cv2
import json

# --- Parámetros de Q-learning OPTIMIZADOS ---
ALPHA = 0.3
GAMMA = 0.95
EPSILON = 0.9
EPSILON_DECAY = 0.996
EPSILON_MIN = 0.05
LEARNING_START = 10
SAVE_INTERVAL = 25

#  ACCIONES DISCRETAS Y ATÓMICAS
ACTIONS = [
    "move 1",  # 0 → Avanzar hasta próximo punto de decisión
    "turn 1.0",  # 1 → Giro de 90° a derecha (discreto)
    "turn -1.0",  # 2 → Giro de 90° a izquierda (discreto)
    "move 0",  # 3 → Detener (para transiciones)
    "turn 0",  # 4 → Detener giro
]

#  TIEMPOS ESTIMADOS PARA ACCIONES (en steps)
ACTION_DURATIONS = {
    0: 15,  # Avanzar un segmento (aumentado)
    1: 12,  # Giro completo 90° derecha
    2: 12,  # Giro completo 90° izquierda
    3: 1,  # Detener
    4: 1,  # Detener giro
}

Q = {}

#  COORDENADAS CORREGIDAS para laberinto 5x5 con escala 3
CHECKPOINTS = {
    "start": {"pos": (1.5, 1.5), "collected": False, "reward": 5},
    "gold_block": {"pos": (7.5, 7.5), "collected": False, "reward": 50},
    "redstone_block": {"pos": (13.5, 13.5), "collected": False, "reward": 200}
}

VISITED_POSITIONS = set()
POSITION_HISTORY = deque(maxlen=50)
LAST_ACTIONS = deque(maxlen=8)

# Estadísticas globales
episode_rewards = []
episode_steps = []
completion_times = []
best_reward = -float('inf')


class DiscreteMovementManager:
    """ Gestor de movimientos discretos y secuenciales"""

    def __init__(self):
        self.current_action = None
        self.action_progress = 0
        self.total_duration = 0
        self.movement_state = "deciding"
        self.consecutive_errors = 0

    def get_action(self, state, epsilon, Q_table, agent_pos=None, info=None):
        front_clear, left_clear, right_clear, far_clear, bright_area = state

        #  RESETEO POR ERRORES CONSECUTIVOS
        if self.consecutive_errors > 5:
            print(" Reseteo por errores consecutivos")
            self.current_action = None
            self.consecutive_errors = 0
            return random.choice([0, 1, 2])

        #  LÓGICA DE DECISIÓN DISCRETA
        if self.current_action is None or self.action_progress >= self.total_duration:
            self.action_progress = 0

            if random.random() < epsilon:
                #  EXPLORACIÓN INTELIGENTE Y DISCRETA
                if front_clear:
                    self.current_action = 0  # Avanzar
                    self.total_duration = ACTION_DURATIONS[0]
                    self.movement_state = "advancing"
                    print(" Avanzando")
                else:
                    # Obstáculo al frente - girar
                    if left_clear and right_clear:
                        self.current_action = random.choice([1, 2])
                        self.total_duration = ACTION_DURATIONS[self.current_action]
                        self.movement_state = "turning"
                        print(" Giro aleatorio (ambas libres)")
                    elif left_clear:
                        self.current_action = 2  # Girar izquierda
                        self.total_duration = ACTION_DURATIONS[2]
                        self.movement_state = "turning"
                        print("️ Giro izquierda")
                    elif right_clear:
                        self.current_action = 1  # Girar derecha
                        self.total_duration = ACTION_DURATIONS[1]
                        self.movement_state = "turning"
                        print("️ Giro derecha")
                    else:
                        # Callejón sin salida - girar 180°
                        self.current_action = 1  # Girar derecha
                        self.total_duration = ACTION_DURATIONS[1] * 2
                        self.movement_state = "turning"
                        print(" Giro 180°")
            else:
                # Explotación de Q-table
                if state not in Q_table:
                    Q_table[state] = np.zeros(len(ACTIONS))
                self.current_action = np.argmax(Q_table[state])
                self.total_duration = ACTION_DURATIONS[self.current_action]
                self.movement_state = "executing"

        # Ejecutar la acción actual
        self.action_progress += 1

        #  DETENER AL FINALIZAR LA ACCIÓN
        if self.action_progress >= self.total_duration:
            print(f" Acción {ACTIONS[self.current_action]} completada")
            return 3 if self.current_action == 0 else 4

        return self.current_action


def get_enhanced_state(obs, info=None):
    """ Estado mejorado con detección corregida"""
    try:
        h, w, d = 360, 640, 3

        if obs is None or obs.size == 0:
            print("️ Observación vacía")
            return (0, 0, 0, 0, 0)

        if obs.size != h * w * d:
            print(f"️ Tamaño incorrecto: {obs.size} != {h * w * d}")
            return (0, 0, 0, 0, 0)

        img = obs.reshape(h, w, d)

        #  VERIFICAR SI LA IMAGEN ES VÁLIDA
        if np.mean(img) < 10:
            print(" Imagen demasiado oscura")
            return (0, 0, 0, 0, 0)

        center_h, center_w = h // 2, w // 2

        #  REGIONES CORREGIDAS - MÁS CENTRADAS 

        # 1. Frente inmediato - MÁS CENTRADO
        front_region = img[center_h + 100:center_h + 140, center_w - 30:center_w + 30]
        front_brightness = np.mean(front_region) if front_region.size > 0 else 0
        front_clear = 1 if front_brightness > 70 else 0  #  Umbral realista

        # 2. Izquierda - MÁS CENTRADO
        left_region = img[center_h + 80:center_h + 120, center_w - 80:center_w - 40]
        left_brightness = np.mean(left_region) if left_region.size > 0 else 0
        left_clear = 1 if left_brightness > 65 else 0

        # 3. Derecha - MÁS CENTRADO
        right_region = img[center_h + 80:center_h + 120, center_w + 40:center_w + 80]
        right_brightness = np.mean(right_region) if right_region.size > 0 else 0
        right_clear = 1 if right_brightness > 65 else 0

        # 4. Frente lejano
        far_front_region = img[center_h + 50:center_h + 90, center_w - 50:center_w + 50]
        far_front_brightness = np.mean(far_front_region) if far_front_region.size > 0 else 0
        far_front_clear = 1 if far_front_brightness > 60 else 0

        # 5. Brillo general
        overall_brightness = np.mean(img)
        bright_area = 1 if overall_brightness > 80 else 0

        print(
            f" F={front_brightness:.1f}({front_clear}), L={left_brightness:.1f}({left_clear}), R={right_brightness:.1f}({right_clear})")

        return (front_clear, left_clear, right_clear, far_front_clear, bright_area)

    except Exception as e:
        print(f" Error en get_enhanced_state: {e}")
        return (0, 0, 0, 0, 0)


def improved_rewards(info, done, action_idx, steps, agent_pos, movement_state):
    """ Sistema de recompensas mejorado"""
    reward = 0

    #  DETECCIÓN DE CHECKPOINTS
    if agent_pos:
        for checkpoint_name, checkpoint_data in CHECKPOINTS.items():
            if not checkpoint_data["collected"] and is_on_checkpoint(agent_pos, checkpoint_data["pos"], tolerance=2.5):
                checkpoint_reward = checkpoint_data["reward"]
                reward += checkpoint_reward
                checkpoint_data["collected"] = True
                print(f" ¡{checkpoint_name.upper()} ALCANZADO! +{checkpoint_reward} puntos")

                if checkpoint_name == "redstone_block":
                    done = True
                    reward += 300
                    print(" ¡LABERINTO COMPLETADO!")
                    return reward, done

    # Recompensa base
    reward += 0.01

    #  RECOMPENSAS PARA MOVIMIENTOS
    if movement_state == "advancing" and action_idx == 0:
        reward += 0.3
    elif movement_state == "turning" and action_idx in [1, 2]:
        reward += 0.2

    #  EXPLORACIÓN
    if agent_pos:
        pos_key = (round(agent_pos[0]), round(agent_pos[1]))
        if pos_key not in VISITED_POSITIONS:
            reward += 0.8
            VISITED_POSITIONS.add(pos_key)
            POSITION_HISTORY.append(pos_key)
            print(f"️ Nueva área: {pos_key}")

    # Penalización temporal mínima
    reward -= 0.01

    return reward, done


def is_on_checkpoint(agent_pos, checkpoint_pos, tolerance=2.5):
    """Verifica si el agente está cerca de un checkpoint"""
    if agent_pos is None:
        return False
    agent_x, agent_z = agent_pos
    checkpoint_x, checkpoint_z = checkpoint_pos
    distance = np.sqrt((agent_x - checkpoint_x) ** 2 + (agent_z - checkpoint_z) ** 2)
    return distance < tolerance


def get_agent_position(info):
    """Obtiene la posición del agente desde la info"""
    try:
        if info and isinstance(info, list) and len(info) > 0:
            pos_info = info[0]
            if isinstance(pos_info, dict) and 'XPos' in pos_info and 'ZPos' in pos_info:
                return (pos_info['XPos'], pos_info['ZPos'])
    except:
        pass
    return None


def robust_env_step(env, action, max_retries=2):
    """Ejecuta step con reintentos mejorados"""
    for attempt in range(max_retries):
        try:
            next_obs, reward, done, info = env.step(action)
            if next_obs is not None and next_obs.size > 0 and np.mean(next_obs) > 10:
                return next_obs, reward, done, info
            time.sleep(0.1)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"️ Error en step: {e}")
            time.sleep(0.1)
    # Fallback seguro
    return np.zeros((360, 640, 3), dtype=np.uint8), 0, False, []


def intelligent_agent(obs, epsilon, info=None, action_manager=None):
    """ Agente inteligente mejorado"""
    state = get_enhanced_state(obs, info)
    if state not in Q:
        Q[state] = np.zeros(len(ACTIONS))

    agent_pos = get_agent_position(info)
    action_idx = action_manager.get_action(state, epsilon, Q, agent_pos, info)

    # Log del estado
    if steps % 20 == 0:
        print(f" {action_manager.movement_state}: {ACTIONS[action_idx]}")

    return action_idx, state


def update_Q(state, action_idx, reward, next_state):
    """Actualización de Q-table"""
    if next_state not in Q:
        Q[next_state] = np.zeros(len(ACTIONS))

    best_next_action = np.max(Q[next_state])
    Q[state][action_idx] += ALPHA * (reward + GAMMA * best_next_action - Q[state][action_idx])


def save_checkpoint(episode, total_reward, epsilon):
    """Guardar progreso"""
    checkpoint_data = {
        'Q_table': Q,
        'episode': episode,
        'total_reward': total_reward,
        'epsilon': epsilon,
        'timestamp': time.time(),
        'stats': {
            'episode_rewards': episode_rewards,
            'episode_steps': episode_steps,
            'completion_times': completion_times
        }
    }

    checkpoint_file = f'checkpoint_ep_{episode}.npy'
    np.save(checkpoint_file, checkpoint_data)

    global best_reward
    if total_reward > best_reward:
        best_reward = total_reward
        np.save('best_model.npy', checkpoint_data)
        print(f" Mejor modelo: {total_reward:.1f} puntos")

    print(f" Checkpoint: {checkpoint_file}")


# Inicializar gestor de movimientos discretos
action_manager = DiscreteMovementManager()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Maze Runner - Movimientos Discretos')
    parser.add_argument('--mission', type=str, default='missions/mazerunner5x5.xml')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--server', type=str, default='127.0.0.1')
    parser.add_argument('--episodes', type=int, default=150)
    parser.add_argument('--episodemaxsteps', type=int, default=300)
    parser.add_argument('--load', type=str, help='Checkpoint a cargar')
    parser.add_argument('--debug', action='store_true', help='Modo debug')

    args = parser.parse_args()

    try:
        xml = Path(args.mission).read_text()
        print(f" XML cargado: {args.mission}")
    except Exception as e:
        print(f" Error cargando XML: {e}")
        exit(1)

    env = malmoenv.make()

    try:
        print(" Inicializando entorno...")
        env.init(xml, args.port, server=args.server)

        start_episode = 0
        epsilon = EPSILON

        if args.load:
            try:
                checkpoint = np.load(args.load, allow_pickle=True).item()
                Q = checkpoint.get('Q_table', {})
                start_episode = checkpoint.get('episode', 0)
                epsilon = checkpoint.get('epsilon', EPSILON)
                print(f" Checkpoint cargado desde episodio {start_episode}")
            except Exception as e:
                print(f" Error cargando checkpoint: {e}")

        print(f" Iniciando desde episodio {start_episode + 1}")
        print(f" Parámetros: α={ALPHA}, γ={GAMMA}, ε={epsilon:.3f}")

        completed_episodes = 0
        start_time = time.time()

        for ep in range(start_episode, args.episodes):
            print(f"\n EPISODIO {ep + 1}/{args.episodes}")
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0

            # Reset para nuevo episodio
            VISITED_POSITIONS.clear()
            POSITION_HISTORY.clear()
            LAST_ACTIONS.clear()
            action_manager = DiscreteMovementManager()

            for checkpoint_name in CHECKPOINTS:
                CHECKPOINTS[checkpoint_name]["collected"] = False

            episode_start = time.time()

            while not done and steps < args.episodemaxsteps:
                action_idx, state = intelligent_agent(obs, epsilon, None, action_manager)
                LAST_ACTIONS.append(action_idx)

                try:
                    time.sleep(0.03)

                    next_obs, env_reward, done, info = robust_env_step(env, action_idx)
                    next_state = get_enhanced_state(next_obs, info)

                    agent_pos = get_agent_position(info)
                    custom_reward, custom_done = improved_rewards(info, done, action_idx, steps, agent_pos,
                                                                  action_manager.movement_state)
                    done = done or custom_done

                    if ep >= LEARNING_START:
                        update_Q(state, action_idx, custom_reward, next_state)

                    obs = next_obs
                    steps += 1
                    total_reward += custom_reward

                    if args.debug and steps % 25 == 0:
                        print(f"   Step {steps}: R={custom_reward:.2f}, Total={total_reward:.1f}")
                        if agent_pos:
                            print(f"   Pos: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})")

                    if steps > 60 and total_reward < -20:
                        print("   ⏹️ Early stop - bajo rendimiento")
                        break

                except Exception as e:
                    print(f" Error en step {steps}: {e}")
                    break

                if done:
                    completed_episodes += 1
                    episode_time = time.time() - episode_start
                    completion_times.append(episode_time)
                    print(f"    COMPLETADO en {steps} steps, {episode_time:.1f}s!")
                    break

            # Estadísticas del episodio
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            # Resumen
            status = "" if done else ""
            success_rate = completed_episodes / (ep + 1) * 100
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward

            print(f"   {status} R: {total_reward:6.1f} | Avg: {avg_reward:5.1f} | Steps: {steps:3d}")
            print(f"   ε: {epsilon:.3f} | Success: {success_rate:5.1f}% | Q-states: {len(Q)}")

            if (ep + 1) % SAVE_INTERVAL == 0 or (done and total_reward > 50):
                save_checkpoint(ep + 1, total_reward, epsilon)

        # Estadísticas finales
        total_time = time.time() - start_time
        print(f"\n ENTRENAMIENTO COMPLETADO")
        print(f" Tiempo total: {total_time / 60:.1f} minutos")
        print(f" Episodios completados: {completed_episodes}/{args.episodes}")
        print(f" Mejor recompensa: {best_reward:.1f}")

    except KeyboardInterrupt:
        print("\n⏹️ Entrenamiento interrumpido")
    except Exception as e:
        print(f" Error crítico: {e}")
        import traceback

        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass
        print(" Entorno cerrado")