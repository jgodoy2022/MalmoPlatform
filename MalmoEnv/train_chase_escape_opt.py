"""
Sistema de Entrenamiento Multi-Agente con Stable-Baselines3
PPO (Perseguidor) vs DQN (Escapista) en Malmo

Ventajas de usar SB3:
- Implementaciones optimizadas y testeadas
- Soporte para CNN/MLP automático
- Mejor exploración y convergencia
- Logging integrado con TensorBoard
- Checkpoints automáticos
"""
import malmoenv
import numpy as np
from pathlib import Path
from lxml import etree
from threading import Thread, Lock, Barrier
import time
import gymnasium as gym
from gymnasium import spaces
import json
from collections import deque

# Importar Stable-Baselines3
try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    print("✓ Stable-Baselines3 importado correctamente")
except ImportError:
    print("❌ ERROR: Instala stable-baselines3:")
    print("   pip install stable-baselines3[extra]")
    exit(1)


# ==================== WRAPPER PARA MALMO ====================
class MalmoGymWrapper(gym.Env):
    """
    Wrapper que convierte MalmoEnv a formato Gymnasium/SB3
    """
    def __init__(self, xml, port, server, server2, port2, role, exp_uid):
        super().__init__()
        
        self.role = role
        self.role_name = "Perseguidor" if role == 0 else "Escapista"
        self.xml = xml
        self.port = port
        self.server = server
        self.server2 = server2
        self.port2 = port2
        self.exp_uid = exp_uid
        
        # Definir espacios de observación y acción
        self.observation_space = spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(15,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # N, S, E, O
        
        # Estado interno
        self.env = None
        self.prev_obs = None
        self.episode_steps = 0
        self.max_steps = 500
        self.episode_reward = 0
        
        # Inicializar Malmo
        self._init_malmo()
    
    def _init_malmo(self):
        """Inicializa la conexión con Malmo"""
        try:
            self.env = malmoenv.make()
            self.env.init(
                self.xml,
                self.port,
                server=self.server,
                server2=self.server2,
                port2=self.port2,
                role=self.role,
                exp_uid=self.exp_uid
            )
            print(f"[{self.role_name}] ✓ Conectado al puerto {self.port if self.role == 0 else self.port2}")
        except Exception as e:
            print(f"[{self.role_name}] ✗ Error de conexión: {e}")
            raise
    
    def _extract_state(self, obs):
        """Extrae features del diccionario de observación"""
        features = []
        
        if not isinstance(obs, dict):
            return np.zeros(15, dtype=np.float32)
        
        # 1. Posición propia (normalizada)
        my_x = obs.get('XPos', 0.0)
        my_z = obs.get('ZPos', 0.0)
        features.extend([my_x / 10.0, my_z / 10.0])
        
        # 2. Velocidad (si está disponible)
        my_vx = obs.get('XVel', 0.0)
        my_vz = obs.get('ZVel', 0.0)
        features.extend([my_vx, my_vz])
        
        # 3. Yaw (orientación normalizada)
        yaw = obs.get('Yaw', 0.0) / 180.0
        features.append(yaw)
        
        # 4. Buscar al enemigo
        enemy_found = False
        entities = obs.get('entities', [])
        if isinstance(entities, list):
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                name = str(entity.get('name', ''))
                # Buscar al otro agente
                if 'Agent' in name or 'Perseguidor' in name or 'Escapista' in name:
                    enemy_x = entity.get('x', my_x)
                    enemy_z = entity.get('z', my_z)
                    enemy_yaw = entity.get('yaw', 0.0)
                    
                    # Posición relativa
                    rel_x = (enemy_x - my_x) / 20.0
                    rel_z = (enemy_z - my_z) / 20.0
                    
                    # Distancia
                    distance = np.sqrt(rel_x**2 + rel_z**2)
                    
                    # Ángulo hacia el enemigo
                    angle = np.arctan2(rel_z, rel_x) / np.pi
                    
                    # Velocidad relativa (si está disponible)
                    enemy_vx = entity.get('motionX', 0.0)
                    enemy_vz = entity.get('motionZ', 0.0)
                    rel_vx = (enemy_vx - my_vx) * 10.0
                    rel_vz = (enemy_vz - my_vz) * 10.0
                    
                    features.extend([rel_x, rel_z, distance, angle, rel_vx, rel_vz])
                    enemy_found = True
                    break
        
        if not enemy_found:
            features.extend([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
        
        # 5. Distancia a bordes del mundo
        border_dist_x = (10.0 - abs(my_x)) / 10.0
        border_dist_z = (10.0 - abs(my_z)) / 10.0
        features.extend([border_dist_x, border_dist_z])
        
        # Asegurar tamaño exacto
        features = features[:15]
        while len(features) < 15:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, obs, prev_obs):
        """Calcula recompensa personalizada según rol"""
        reward = 0.0
        
        if not isinstance(obs, dict) or not isinstance(prev_obs, dict):
            return reward
        
        my_x = obs.get('XPos', 0.0)
        my_z = obs.get('ZPos', 0.0)
        prev_my_x = prev_obs.get('XPos', my_x)
        prev_my_z = prev_obs.get('ZPos', my_z)
        
        # Penalización por movimiento (eficiencia)
        movement_penalty = -0.01
        
        # Buscar distancia al enemigo
        enemy_dist = None
        prev_enemy_dist = None
        
        entities = obs.get('entities', [])
        if isinstance(entities, list):
            for entity in entities:
                if isinstance(entity, dict):
                    name = str(entity.get('name', ''))
                    if 'Agent' in name or 'Perseguidor' in name or 'Escapista' in name:
                        enemy_x = entity.get('x', my_x)
                        enemy_z = entity.get('z', my_z)
                        enemy_dist = np.sqrt((my_x - enemy_x)**2 + (my_z - enemy_z)**2)
                        break
        
        prev_entities = prev_obs.get('entities', [])
        if isinstance(prev_entities, list):
            for entity in prev_entities:
                if isinstance(entity, dict):
                    name = str(entity.get('name', ''))
                    if 'Agent' in name or 'Perseguidor' in name or 'Escapista' in name:
                        enemy_x = entity.get('x', prev_my_x)
                        enemy_z = entity.get('z', prev_my_z)
                        prev_enemy_dist = np.sqrt((prev_my_x - enemy_x)**2 + (prev_my_z - enemy_z)**2)
                        break
        
        # Recompensas específicas por rol
        if self.role == 0:  # PERSEGUIDOR
            if enemy_dist is not None:
                # Recompensa por acercarse
                if prev_enemy_dist is not None:
                    delta = prev_enemy_dist - enemy_dist
                    reward += delta * 2.0  # Multiplicador fuerte
                
                # Bonificaciones por distancia
                if enemy_dist < 1.5:
                    reward += 100.0  # ¡CAPTURA!
                elif enemy_dist < 2.5:
                    reward += 5.0   # Muy cerca
                elif enemy_dist < 4.0:
                    reward += 1.0   # Cerca
                else:
                    reward -= 0.5   # Lejos, penalización
        
        else:  # ESCAPISTA (role 1)
            # Recompensa por supervivencia
            reward += 0.2
            
            if enemy_dist is not None:
                # Recompensa por alejarse
                if prev_enemy_dist is not None:
                    delta = enemy_dist - prev_enemy_dist
                    reward += delta * 2.0
                
                # Penalizaciones/bonificaciones por distancia
                if enemy_dist < 1.5:
                    reward -= 100.0  # ¡CAPTURADO!
                elif enemy_dist < 2.5:
                    reward -= 5.0    # Peligro
                elif enemy_dist < 4.0:
                    reward -= 1.0    # Cerca
                else:
                    reward += 1.0    # Seguro, bien!
        
        # Penalización por salirse del mundo
        if abs(my_x) > 9.5 or abs(my_z) > 9.5:
            reward -= 10.0
        
        return reward + movement_penalty
    
    def reset(self, seed=None, options=None):
        """Reset del entorno"""
        super().reset(seed=seed)
        
        try:
            obs = self.env.reset()
            self.prev_obs = obs if isinstance(obs, dict) else {}
            self.episode_steps = 0
            self.episode_reward = 0
            
            state = self._extract_state(obs)
            return state, {}
        
        except Exception as e:
            print(f"[{self.role_name}] Error en reset: {e}")
            return np.zeros(15, dtype=np.float32), {}
    
    def step(self, action):
        """Ejecuta una acción"""
        try:
            # Ejecutar acción en Malmo
            obs, malmo_reward, done, info = self.env.step(action)
            
            # Calcular recompensa personalizada
            custom_reward = self._calculate_reward(obs, self.prev_obs)
            total_reward = malmo_reward + custom_reward
            
            # Extraer estado
            state = self._extract_state(obs)
            
            # Actualizar
            self.prev_obs = obs if isinstance(obs, dict) else {}
            self.episode_steps += 1
            self.episode_reward += total_reward
            
            # Terminar si se excede el máximo de pasos
            if self.episode_steps >= self.max_steps:
                done = True
                info['TimeLimit.truncated'] = True
            
            return state, total_reward, done, False, info
        
        except Exception as e:
            print(f"[{self.role_name}] Error en step: {e}")
            return np.zeros(15, dtype=np.float32), -1.0, True, False, {}
    
    def close(self):
        """Cierra el entorno"""
        if self.env:
            try:
                self.env.close()
            except:
                pass


# ==================== CALLBACKS PERSONALIZADOS ====================
class MultiAgentCallback(BaseCallback):
    """Callback para logging y estadísticas durante entrenamiento"""
    def __init__(self, role, role_name, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.role = role
        self.role_name = role_name
        self.check_freq = check_freq
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        # Capturar info de episodios completados
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
        
        # Logging periódico
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards)
                mean_length = np.mean(self.episode_lengths)
                
                print(f"\n[{self.role_name}] Paso {self.n_calls}")
                print(f"  Reward promedio (100 eps): {mean_reward:.2f}")
                print(f"  Longitud promedio: {mean_length:.1f}")
                
                # Guardar mejor modelo
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    model_path = f"models/{self.role_name.lower()}_best.zip"
                    self.model.save(model_path)
                    print(f"  ✓ Nuevo mejor modelo guardado: {mean_reward:.2f}")
        
        return True


# ==================== ENTRENAMIENTO POR AGENTE ====================
def train_agent_sb3(role, xml, port, server, server2, total_timesteps, start_barrier):
    """
    Entrena un agente usando Stable-Baselines3
    """
    role_name = "Perseguidor_PPO" if role == 0 else "Escapista_DQN"
    
    print(f"\n[{role_name}] Inicializando...")
    
    # Esperar a que todos los agentes estén listos
    print(f"[{role_name}] Esperando sincronización...")
    start_barrier.wait()
    time.sleep(role * 2)  # Escalonar inicio para evitar conflictos
    
    try:
        # Crear entorno
        env = MalmoGymWrapper(
            xml=xml,
            port=port,
            server=server,
            server2=server2,
            port2=port + 1,
            role=role,
            exp_uid='sb3_multiagent_training'
        )
        
        # Wrap con Monitor para estadísticas
        env = Monitor(env, filename=f"logs/{role_name}")
        
        # Crear modelo según rol
        if role == 0:  # PERSEGUIDOR - PPO
            print(f"[{role_name}] Creando modelo PPO...")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
                tensorboard_log=f"./tensorboard/{role_name}/"
            )
        
        else:  # ESCAPISTA - DQN
            print(f"[{role_name}] Creando modelo DQN...")
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=1e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                exploration_fraction=0.3,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                verbose=1,
                tensorboard_log=f"./tensorboard/{role_name}/"
            )
        
        # Crear callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"./models/{role_name}/",
            name_prefix=role_name
        )
        
        custom_callback = MultiAgentCallback(
            role=role,
            role_name=role_name,
            check_freq=1000
        )
        
        # ENTRENAR
        print(f"\n[{role_name}] 🚀 Iniciando entrenamiento ({total_timesteps} steps)...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, custom_callback],
            progress_bar=True
        )
        
        # Guardar modelo final
        final_path = f"models/{role_name}_FINAL.zip"
        model.save(final_path)
        print(f"\n[{role_name}] ✓ Modelo final guardado: {final_path}")
        
        env.close()
        print(f"[{role_name}] ✓ Entrenamiento completado!")
    
    except Exception as e:
        print(f"[{role_name}] ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


# ==================== MAIN ====================
if __name__ == '__main__':
    print("=" * 70)
    print("  ENTRENAMIENTO MULTI-AGENTE CON STABLE-BASELINES3")
    print("  PPO (Perseguidor) vs DQN (Escapista)")
    print("=" * 70)
    
    # Crear carpetas
    Path('models').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    Path('tensorboard').mkdir(exist_ok=True)
    
    # Cargar XML
    xml_path = Path('missions/chase_escape.xml')
    if not xml_path.exists():
        print(f"\n❌ ERROR: No se encuentra {xml_path}")
        print("Asegúrate de tener el archivo XML de la misión")
        exit(1)
    
    xml = xml_path.read_text()
    
    # Verificar agentes
    mission = etree.fromstring(xml)
    number_of_agents = len(mission.findall('{http://ProjectMalmo.microsoft.com}AgentSection'))
    
    if number_of_agents != 2:
        print(f"\n❌ ERROR: La misión debe tener 2 agentes (encontrados: {number_of_agents})")
        exit(1)
    
    print(f"\n✓ Misión cargada: {number_of_agents} agentes")
    
    # Configuración
    PORT = 9000
    SERVER = '127.0.0.1'
    SERVER2 = SERVER
    TOTAL_TIMESTEPS = 500000  # 500k steps de entrenamiento
    
    print(f"\nConfiguración:")
    print(f"  Puerto base: {PORT}")
    print(f"  Agente 0 (Perseguidor-PPO): puerto {PORT}")
    print(f"  Agente 1 (Escapista-DQN): puerto {PORT + 1}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Algoritmos: PPO vs DQN")
    
    # Instrucciones
    print("\n" + "=" * 70)
    print("INSTRUCCIONES:")
    print("  1. Abre DOS terminales adicionales")
    print(f"  2. Terminal 1: py -3.7 -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft({PORT})\"")
    print(f"  3. Terminal 2: py -3.7 -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft({PORT + 1})\"")
    print("  4. Espera a que AMBAS instancias muestren 'SERVER STARTED'")
    print("=" * 70)
    print("\nPresiona ENTER cuando ambas instancias estén listas...")
    input()
    
    print("\n🚀 Iniciando entrenamiento en 3 segundos...")
    time.sleep(3)
    
    # Barrier para sincronizar
    start_barrier = Barrier(number_of_agents)
    
    # Crear threads
    threads = [
        Thread(
            target=train_agent_sb3,
            args=(i, xml, PORT, SERVER, SERVER2, TOTAL_TIMESTEPS, start_barrier),
            name=f"Agent-{i}"
        )
        for i in range(number_of_agents)
    ]
    
    # Iniciar
    [t.start() for t in threads]
    [t.join() for t in threads]
    
    # Resumen final
    print("\n" + "=" * 70)
    print("  ✓ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print("\nModelos guardados en:")
    print("  - models/Perseguidor_PPO_FINAL.zip")
    print("  - models/Escapista_DQN_FINAL.zip")
    print("\nPara visualizar el entrenamiento:")
    print("  tensorboard --logdir ./tensorboard")
    print("\n✅ ¡Entrenamiento exitoso!")