"""
Sistema de Entrenamiento Multi-Agente: Perseguidor vs Escapista
PPO (Perseguidor) vs DQN (Escapista) en Malmo
Basado en la estructura de runmultiagent.py
"""
import malmoenv
import numpy as np
from pathlib import Path
from lxml import etree
from threading import Thread, Lock
import time
from collections import deque
import random
import json

# ==================== AGENTE DQN (ESCAPISTA) ====================
class DQNAgent:
    """Deep Q-Network para el agente escapista"""
    def __init__(self, state_size, action_size, agent_id="DQN_Escapista"):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        
        # Hiperparámetros
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=5000)
        self.batch_size = 64
        
        # Redes Q
        self.q_network = np.random.randn(state_size, action_size) * 0.01
        self.target_network = self.q_network.copy()
        self.update_target_counter = 0
        self.update_target_freq = 100
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = np.dot(state, self.q_network)
        return np.argmax(q_values)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        minibatch = random.sample(self.memory, self.batch_size)
        total_loss = 0.0
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q = np.dot(next_state, self.target_network)
                target = reward + self.gamma * np.max(next_q)
            
            current_q = np.dot(state, self.q_network)
            td_error = target - current_q[action]
            self.q_network[:, action] += self.learning_rate * td_error * state
            total_loss += td_error ** 2
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.update_target_counter += 1
        if self.update_target_counter >= self.update_target_freq:
            self.target_network = self.q_network.copy()
            self.update_target_counter = 0
        
        return total_loss / self.batch_size
    
    def save(self, filename):
        np.save(filename, {'q_net': self.q_network, 'epsilon': self.epsilon})
    
    def load(self, filename):
        data = np.load(filename, allow_pickle=True).item()
        self.q_network = data['q_net']
        self.epsilon = data['epsilon']
        self.target_network = self.q_network.copy()


# ==================== AGENTE PPO (PERSEGUIDOR) ====================
class PPOAgent:
    """Proximal Policy Optimization para el agente perseguidor"""
    def __init__(self, state_size, action_size, agent_id="PPO_Perseguidor"):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        
        # Hiperparámetros
        self.gamma = 0.95
        self.lambda_gae = 0.95
        self.epsilon_clip = 0.2
        self.learning_rate = 0.0003
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.epochs = 10
        
        # Redes
        self.actor_weights = np.random.randn(state_size, action_size) * 0.01
        self.critic_weights = np.random.randn(state_size, 1) * 0.01
        
        # Buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def get_action_probs(self, state):
        logits = np.dot(state, self.actor_weights)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-8)
        return probs
    
    def get_value(self, state):
        return np.dot(state, self.critic_weights).item()
    
    def act(self, state):
        probs = self.get_action_probs(state)
        action = np.random.choice(self.action_size, p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        value = self.get_value(state)
        
        self.states.append(state.copy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        return action
    
    def store_outcome(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self):
        if len(self.states) < 5:
            return 0.0
        
        returns = []
        advantages = []
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
            next_value = self.values[t]
        
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = np.array(returns)
        
        total_loss = 0.0
        for epoch in range(self.epochs):
            for i in range(len(self.states)):
                state = self.states[i]
                action = self.actions[i]
                advantage = advantages[i]
                return_val = returns[i]
                
                self.actor_weights[:, action] += self.learning_rate * advantage * state
                
                value_pred = self.get_value(state)
                value_error = return_val - value_pred
                self.critic_weights += self.learning_rate * value_error * state.reshape(-1, 1)
        
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        
        return total_loss
    
    def save(self, filename):
        np.savez(filename, actor=self.actor_weights, critic=self.critic_weights)
    
    def load(self, filename):
        data = np.load(filename)
        self.actor_weights = data['actor']
        self.critic_weights = data['critic']


# ==================== EXTRACCIÓN DE ESTADO ====================
def extract_state_from_obs(obs):
    """Extrae features del diccionario de observación"""
    features = []
    
    # Posición propia
    my_x = obs.get('XPos', 0.0)
    my_z = obs.get('ZPos', 0.0)
    features.extend([my_x / 6.0, my_z / 6.0])
    
    # Buscar al otro agente
    enemy_found = False
    entities = obs.get('entities', [])
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, dict):
                name = entity.get('name', '')
                if 'Agent' in str(name) or 'Perseguidor' in str(name) or 'Escapista' in str(name):
                    enemy_x = entity.get('x', 0.0)
                    enemy_z = entity.get('z', 0.0)
                    rel_x = (enemy_x - my_x) / 12.0
                    rel_z = (enemy_z - my_z) / 12.0
                    distance = np.sqrt(rel_x**2 + rel_z**2)
                    angle = np.arctan2(rel_z, rel_x) / np.pi
                    features.extend([rel_x, rel_z, distance, angle])
                    enemy_found = True
                    break
    
    if not enemy_found:
        features.extend([0.0, 0.0, 1.0, 0.0])
    
    # Obstáculos (simplificado)
    features.extend([0, 0, 0, 0])
    
    # Distancia a paredes
    features.extend([(6.0 - abs(my_x)) / 6.0, (6.0 - abs(my_z)) / 6.0])
    
    # Asegurar tamaño fijo
    while len(features) < 15:
        features.append(0.0)
    
    return np.array(features[:15], dtype=np.float32)


def calculate_custom_reward(obs, prev_obs, role):
    """Calcula recompensas adicionales"""
    reward = 0.0
    
    if not isinstance(obs, dict) or not isinstance(prev_obs, dict):
        return reward
    
    my_x = obs.get('XPos', 0.0)
    my_z = obs.get('ZPos', 0.0)
    
    # Buscar distancia al enemigo
    enemy_distance = None
    prev_enemy_distance = None
    
    entities = obs.get('entities', [])
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, dict):
                name = entity.get('name', '')
                if 'Agent' in str(name) or 'Perseguidor' in str(name) or 'Escapista' in str(name):
                    enemy_x = entity.get('x', my_x)
                    enemy_z = entity.get('z', my_z)
                    enemy_distance = np.sqrt((my_x - enemy_x)**2 + (my_z - enemy_z)**2)
                    break
    
    prev_entities = prev_obs.get('entities', [])
    prev_my_x = prev_obs.get('XPos', my_x)
    prev_my_z = prev_obs.get('ZPos', my_z)
    if isinstance(prev_entities, list):
        for entity in prev_entities:
            if isinstance(entity, dict):
                name = entity.get('name', '')
                if 'Agent' in str(name) or 'Perseguidor' in str(name) or 'Escapista' in str(name):
                    enemy_x = entity.get('x', prev_my_x)
                    enemy_z = entity.get('z', prev_my_z)
                    prev_enemy_distance = np.sqrt((prev_my_x - enemy_x)**2 + (prev_my_z - enemy_z)**2)
                    break
    
    # Recompensas según rol
    if role == 0:  # Perseguidor
        if enemy_distance is not None:
            if prev_enemy_distance is not None:
                if enemy_distance < prev_enemy_distance:
                    reward += 0.5
                else:
                    reward -= 0.2
            if enemy_distance < 1.5:
                reward += 50.0
            elif enemy_distance < 3.0:
                reward += 2.0
    else:  # Escapista
        if enemy_distance is not None:
            if prev_enemy_distance is not None:
                if enemy_distance > prev_enemy_distance:
                    reward += 0.5
                else:
                    reward -= 0.2
            if enemy_distance < 1.5:
                reward -= 50.0
            elif enemy_distance < 3.0:
                reward -= 2.0
            reward += 0.1  # Bonus por sobrevivir
    
    return reward


# ==================== FUNCIÓN DE ENTRENAMIENTO POR AGENTE ====================
def train_agent(role, xml, port, server, server2, episodes, agents_dict, stats_lock, global_stats):
    """
    Función de entrenamiento para cada agente en su thread
    Usa la misma lógica de conexión que runmultiagent.py
    """
    role_name = "PERSEGUIDOR-PPO" if role == 0 else "ESCAPISTA-DQN"
    
    print(f"\n[{role_name}] Inicializando conexión...")
    
    # Inicializar entorno EXACTAMENTE como runmultiagent.py
    try:
        env = malmoenv.make()
        env.init(xml,
                 port, server=server,
                 server2=server2, port2=(port + role),  # ¡CLAVE! port2 = port + role
                 role=role,
                 exp_uid='ppo_vs_dqn_training')
        print(f"[{role_name}] ✅ Conexión establecida en puerto {port + role}")
    except Exception as e:
        print(f"[{role_name}] ❌ ERROR al conectar: {e}")
        return
    
    agent = agents_dict[role]
    print(f"[{role_name}] Iniciando entrenamiento...")
    
    for episode in range(episodes):
        print(f"[{role_name}] Episodio {episode + 1}/{episodes}")
        
        try:
            obs = env.reset()
        except Exception as e:
            print(f"[{role_name}] Error en reset: {e}")
            time.sleep(2)
            continue
        
        # Inicializar observación previa
        prev_obs = {'XPos': 0.0, 'ZPos': 0.0, 'entities': []}
        if isinstance(obs, dict):
            prev_obs = obs.copy()
        
        done = False
        episode_reward = 0
        steps = 0
        episode_loss = 0
        
        while not done and steps < 500:
            try:
                # Extraer estado
                if isinstance(obs, dict):
                    state = extract_state_from_obs(obs)
                else:
                    state = np.zeros(15, dtype=np.float32)
                
                # Seleccionar acción
                action = agent.act(state)
                
                # Ejecutar acción
                obs, reward, done, info = env.step(action)
                
                # Extraer siguiente estado
                if isinstance(obs, dict):
                    next_state = extract_state_from_obs(obs)
                else:
                    next_state = np.zeros(15, dtype=np.float32)
                
                # Calcular recompensa personalizada
                custom_reward = calculate_custom_reward(obs, prev_obs, role)
                total_reward = reward + custom_reward
                
                # Almacenar experiencia según tipo de agente
                if role == 0:  # PPO
                    agent.store_outcome(total_reward, done)
                else:  # DQN
                    agent.remember(state, action, total_reward, next_state, done)
                
                # Actualizar
                if isinstance(obs, dict):
                    prev_obs = obs.copy()
                episode_reward += total_reward
                steps += 1
                
                time.sleep(0.01)
            
            except Exception as e:
                print(f"[{role_name}] Error en step {steps}: {e}")
                break
        
        # Entrenar agente al final del episodio
        if role == 0:  # PPO
            loss = agent.update()
            episode_loss = loss
        else:  # DQN
            # Entrenar varias veces por episodio
            for _ in range(5):
                loss = agent.replay()
                episode_loss += loss
        
        # Guardar estadísticas
        with stats_lock:
            global_stats[role]['episodes'].append(episode + 1)
            global_stats[role]['rewards'].append(episode_reward)
            global_stats[role]['steps'].append(steps)
            global_stats[role]['losses'].append(episode_loss)
            
            # Calcular promedio de últimos 10 episodios
            recent_rewards = global_stats[role]['rewards'][-10:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            global_stats[role]['avg_rewards'].append(avg_reward)
        
        # Log de progreso
        if (episode + 1) % 5 == 0:
            print(f"[{role_name}] Ep {episode + 1}/{episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Steps: {steps} | "
                  f"Avg(10): {avg_reward:.2f} | "
                  f"Loss: {episode_loss:.4f}")
            
            if role == 1:  # DQN
                print(f"  Epsilon: {agent.epsilon:.3f}")
        
        # Guardar checkpoints
        if (episode + 1) % 20 == 0:
            if role == 0:
                agent.save(f'models/ppo_perseguidor_ep{episode + 1}.npz')
            else:
                agent.save(f'models/dqn_escapista_ep{episode + 1}.npy')
            
            # Guardar estadísticas
            with stats_lock:
                with open(f'stats_{role_name}_ep{episode + 1}.json', 'w') as f:
                    json.dump(global_stats[role], f, indent=2)
    
    env.close()
    print(f"\n[{role_name}] Entrenamiento completado!")


# ==================== MAIN ====================
if __name__ == '__main__':
    print("=" * 70)
    print("  ENTRENAMIENTO PPO vs DQN")
    print("=" * 70)
    
    # Crear carpeta models
    Path('models').mkdir(exist_ok=True)
    
    # Cargar XML
    xml_path = Path('missions/chase_escape.xml')
    if not xml_path.exists():
        print(f"ERROR: No se encuentra {xml_path}")
        exit(1)
    
    xml = xml_path.read_text()
    
    # Verificar número de agentes
    mission = etree.fromstring(xml)
    number_of_agents = len(mission.findall('{http://ProjectMalmo.microsoft.com}AgentSection'))
    print(f"Número de agentes: {number_of_agents}")
    
    if number_of_agents != 2:
        print("ERROR: La misión debe tener exactamente 2 agentes")
        exit(1)
    
    # Configuración de conexión (igual que runmultiagent.py)
    STATE_SIZE = 15
    ACTION_SIZE = 4
    EPISODES = 100
    PORT = 9000
    SERVER = '127.0.0.1'
    SERVER2 = SERVER  # Mismo servidor para ambos agentes
    
    print(f"\nConfiguración:")
    print(f"  Estado: {STATE_SIZE} features")
    print(f"  Acciones: {ACTION_SIZE} (N, S, E, O)")
    print(f"  Episodios: {EPISODES}")
    print(f"  Puerto base: {PORT}")
    print(f"  Agente 0 usará puerto: {PORT}")
    print(f"  Agente 1 usará puerto: {PORT + 1}")
    
    # Crear agentes
    print("\nCreando agentes...")
    ppo_agent = PPOAgent(STATE_SIZE, ACTION_SIZE, "PPO_Perseguidor")
    dqn_agent = DQNAgent(STATE_SIZE, ACTION_SIZE, "DQN_Escapista")
    
    agents_dict = {
        0: ppo_agent,  # Role 0: Perseguidor
        1: dqn_agent   # Role 1: Escapista
    }
    
    # Estadísticas compartidas
    stats_lock = Lock()
    global_stats = {
        0: {'episodes': [], 'rewards': [], 'steps': [], 'losses': [], 'avg_rewards': []},
        1: {'episodes': [], 'rewards': [], 'steps': [], 'losses': [], 'avg_rewards': []}
    }
    
    print("\n" + "=" * 70)
    print("ANTES DE CONTINUAR:")
    print("  1. Abre DOS terminales")
    print("  2. Terminal 1: py -3.7 -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9000)\"")
    print("  3. Terminal 2: py -3.7 -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9001)\"")
    print("  4. Espera a que AMBAS instancias estén completamente cargadas")
    print("=" * 70)
    print("\nPresiona ENTER cuando ambas instancias de Minecraft estén listas...")
    input()
    
    print("\n🚀 Iniciando entrenamiento...\n")
    time.sleep(2)
    
    # Crear threads - MISMA LÓGICA que runmultiagent.py
    threads = [
        Thread(target=train_agent, args=(
            i,              # role
            xml,            # xml
            PORT,           # port
            SERVER,         # server
            SERVER2,        # server2
            EPISODES,       # episodes
            agents_dict,    # agents_dict
            stats_lock,     # stats_lock
            global_stats    # global_stats
        )) for i in range(number_of_agents)
    ]
    
    # Iniciar todos los threads simultáneamente (como runmultiagent.py)
    [t.start() for t in threads]
    [t.join() for t in threads]
    
    # Resumen final
    print("\n" + "=" * 70)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    
    # Guardar modelos finales
    ppo_agent.save('models/ppo_perseguidor_FINAL.npz')
    dqn_agent.save('models/dqn_escapista_FINAL.npy')
    
    # Estadísticas finales
    print("\n--- Estadísticas Finales ---")
    for role in [0, 1]:
        role_name = "PERSEGUIDOR-PPO" if role == 0 else "ESCAPISTA-DQN"
        stats = global_stats[role]
        
        if stats['rewards']:
            print(f"\n{role_name}:")
            print(f"  Recompensa promedio: {np.mean(stats['rewards']):.2f}")
            print(f"  Mejor recompensa: {np.max(stats['rewards']):.2f}")
            print(f"  Peor recompensa: {np.min(stats['rewards']):.2f}")
            print(f"  Pasos promedio: {np.mean(stats['steps']):.2f}")
    
    print(f"\nModelos guardados en: ./models/")
    print(f"Estadísticas guardadas en: ./stats_*.json")
    print("\n✅ Entrenamiento finalizado exitosamente!")