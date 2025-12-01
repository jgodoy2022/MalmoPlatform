"""
Sistema de Entrenamiento Multi-Agente CORREGIDO
PPO (Perseguidor) vs DQN (Escapista) en Malmo

FIX CRÍTICO: Las observaciones están en info, no en obs
obs = imagen (numpy array)
info = diccionario JSON con posiciones/entidades
"""
import malmoenv
import numpy as np
from pathlib import Path
from lxml import etree
from threading import Thread, Barrier
import time
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import os
import json

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    print("✓ Stable-Baselines3 importado correctamente")
except ImportError:
    print("❌ ERROR: Instala stable-baselines3")
    exit(1)


# ==================== WRAPPER CORREGIDO ====================
class MalmoGymWrapper(gym.Env):
    """Wrapper CORREGIDO - Extrae datos de info en lugar de obs"""
    
    def __init__(self, xml, port, server, server2, port2, role, exp_uid):
        super().__init__()
        
        self.role = role
        self.role_name = "Perseguidor_PPO" if role == 0 else "Escapista_DQN"
        self.xml = xml
        self.port = port
        self.server = server
        self.server2 = server2
        self.port2 = port2
        self.exp_uid = exp_uid
        
        # Espacios
        self.observation_space = spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(15,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        
        # Estado interno
        self.env = None
        self.prev_info = None  # Cambiado: guardamos info, no obs
        self.episode_steps = 0
        self.max_steps = 500
        self.episode_reward = 0
        
        # Tracking
        self.min_distance_achieved = float('inf')
        self.consecutive_approaches = 0
        self.consecutive_retreats = 0
        
        self._init_malmo()
    
    def _init_malmo(self):
        """Inicializa Malmo"""
        try:
            self.env = malmoenv.make()
            self.env.init(
                self.xml, self.port, server=self.server,
                server2=self.server2, port2=self.port2,
                role=self.role, exp_uid=self.exp_uid
            )
            print(f"[{self.role_name}] ✓ Conectado")
        except Exception as e:
            print(f"[{self.role_name}] ✗ Error: {e}")
            raise
    
    def _parse_info(self, info):
        """
        Parsea info - puede venir como string JSON, dict, o None
        """
        if info is None:
            return {}
        
        if isinstance(info, str):
            try:
                return json.loads(info)
            except json.JSONDecodeError:
                # Si no es JSON válido, intentar evaluar como dict
                try:
                    return eval(info)
                except:
                    return {}
        elif isinstance(info, dict):
            return info
        else:
            return {}
    
    def _extract_state(self, info_dict):
        """
        CORREGIDO: Usa caché cuando Malmo pierde al enemigo.
        """
        features = []

        if not isinstance(info_dict, dict):
            return np.zeros(15, dtype=np.float32)

        # 1. Posición propia (normalizada)
        my_x = info_dict.get('XPos', 0.0)
        my_z = info_dict.get('ZPos', 0.0)
        features.extend([my_x / 10.0, my_z / 10.0])

        # 2. Velocidad (generalmente no disponible directamente, usar 0)
        my_vx = 0.0
        my_vz = 0.0
        features.extend([my_vx, my_vz])

        # 3. Yaw (orientación normalizada)
        yaw = info_dict.get('Yaw', 0.0) / 180.0
        features.append(yaw)

        # 4. Buscar al enemigo en entities
        enemy_found = False
        entities = info_dict.get('entities', [])
        enemy_data = None

        if isinstance(entities, list):
            for entity in entities:
                if not isinstance(entity, dict):
                    continue

                name = str(entity.get('name', ''))
                is_other_agent = (
                    ('Agent' in name) or 
                    ('Perseguidor' in name) or 
                    ('Escapista' in name)
                ) and name != info_dict.get('Name', '')

                if is_other_agent:
                    enemy_data = entity
                    enemy_found = True
                    break

        # ================================
        # FIX: Si no se detecta enemigo
        # ================================
        if not enemy_found:

            # FIX 1: Primer tick del episodio (prev_info=None)
            if self.prev_info is None:
                if info_dict.get("Name") == "Perseguidor_PPO":
                    enemy_x, enemy_z = 4.5, 4.5
                else:
                    enemy_x, enemy_z = -4.5, -4.5

                enemy_data = {"x": enemy_x, "z": enemy_z}

            # FIX 2: Paso normal → usar caché
            elif "last_enemy" in self.prev_info:
                enemy_data = self.prev_info["last_enemy"]

            # FIX 3: No hay nada
            else:
                enemy_data = None

        # =================================

        if enemy_data:
            enemy_x = enemy_data.get('x', my_x)
            enemy_z = enemy_data.get('z', my_z)

            # Guardar última posición válida
            if self.prev_info is None:
                self.prev_info = {}
            self.prev_info["last_enemy"] = {"x": enemy_x, "z": enemy_z}

            rel_x = np.clip((enemy_x - my_x) / 20.0, -1, 1)
            rel_z = np.clip((enemy_z - my_z) / 20.0, -1, 1)

            distance = np.sqrt((enemy_x - my_x)**2 + (enemy_z - my_z)**2) / 10.0
            distance = np.clip(distance, 0, 2)

            angle = np.arctan2(enemy_z - my_z, enemy_x - my_x) / np.pi

            enemy_vx = enemy_data.get('motionX', 0.0)
            enemy_vz = enemy_data.get('motionZ', 0.0)
            rel_vx = np.clip((enemy_vx - my_vx) * 10.0, -1, 1)
            rel_vz = np.clip((enemy_vz - my_vz) * 10.0, -1, 1)

            features.extend([rel_x, rel_z, distance, angle, rel_vx, rel_vz])

        else:
            # Sin enemigo ni caché
            features.extend([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])

        # 5. Distancia a bordes
        border_dist_x = np.clip((10.0 - abs(my_x)) / 10.0, 0, 1)
        border_dist_z = np.clip((10.0 - abs(my_z)) / 10.0, 0, 1)
        features.extend([border_dist_x, border_dist_z])

        # Asegurar tamaño exacto
        features = features[:15]
        while len(features) < 15:
            features.append(0.0)

        return np.array(features, dtype=np.float32)


    
    def _calculate_reward(self, info, prev):
        """
        Recompensas balanceadas
        PPO y DQN en escalas comparables
        """
        if info is None or prev is None:
            return 0.0, False
    
        my_x = info.get("XPos", 0.0)
        my_z = info.get("ZPos", 0.0)
        my_name = info.get("Name", "")

        # Buscar enemigo actual
        enemy_dist = None
        for e in info.get("entities", []):
            if isinstance(e, dict):
                n = e.get("name", "")
                if n != my_name:
                    ex, ez = e.get("x", my_x), e.get("z", my_z)
                    enemy_dist = np.sqrt((my_x-ex)**2 + (my_z-ez)**2)
                    break
    
        # Buscar enemigo previo
        prev_dist = None
        for e in prev.get("entities", []):
            if isinstance(e, dict):
                n = e.get("name", "")
                if n != my_name:
                    px = prev.get("XPos", my_x)
                    pz = prev.get("ZPos", my_z)
                    ex, ez = e.get("x", px), e.get("z", pz)
                    prev_dist = np.sqrt((px-ex)**2 + (pz-ez)**2)
                    break

        # No hay datos suficientes
        if enemy_dist is None or prev_dist is None:
            return 0.0, False

        # Diferencia normalizada de distancia
        delta = prev_dist - enemy_dist
        delta = float(np.clip(delta, -1, 1))

        # -------------------------------
        # REWARD DEL PERSEGUIDOR (PPO)
        # -------------------------------
        if self.role == 0:
            reward = delta * 1.0   # acercarse da +, alejarse da -
        
            # proximidad
            if enemy_dist < 1.3:
                return 3.0, True   # captura
        
            reward -= 0.01         # pequeña penalización por paso

        # -------------------------------
        # REWARD DEL ESCAPISTA (DQN)
        # -------------------------------
        else:
            reward = -delta * 1.0  # alejarse da +, acercarse da -

            # captura
            if enemy_dist < 1.3:
                return -3.0, True

            reward += 0.02         # supervivencia

        return float(reward), False
    
    def reset(self, seed=None, options=None):
        """Reset mejorado"""
        super().reset(seed=seed)
        
        self.min_distance_achieved = float('inf')
        self.consecutive_approaches = 0
        self.consecutive_retreats = 0
        
        try:
            obs = self.env.reset()
            
            # Intentar obtener info del reset
            # Si reset no devuelve info, hacer un step sin acción
            if isinstance(obs, tuple) and len(obs) > 3:
                # Reset devolvió (obs, reward, done, info)
                _, _, _, first_info = obs
                obs = obs[0]
            else:
                # Reset solo devolvió obs, hacer dummy step
                dummy_obs, dummy_reward, dummy_done, first_info = self.env.step(0)
            
            # Parsear info
            info_dict = self._parse_info(first_info)
            
            # Si info está vacío, usar valores por defecto
            if not info_dict or 'XPos' not in info_dict:
                info_dict = {
                    'XPos': -4.5 if self.role == 0 else 4.5,
                    'ZPos': -4.5 if self.role == 0 else 4.5,
                    'Yaw': 45.0 if self.role == 0 else 225.0,
                    'Name': self.role_name,
                    'entities': []
                }
            
            self.prev_info = info_dict
            
            self.episode_steps = 0
            self.episode_reward = 0
            
            state = self._extract_state(info_dict)
            
            # LOGGING INICIAL
            print(f"\n[{self.role_name}] RESET")
            print(f"  Raw info: {first_info}")
            print(f"  Parsed info: {info_dict}")
            print(f"  Initial state: {state}")

            return state, {}
        
        except Exception as e:
            print(f"[{self.role_name}] Error en reset: {e}")
            # Valores por defecto si todo falla
            default_info = {
                'XPos': -4.5 if self.role == 0 else 4.5,
                'ZPos': -4.5 if self.role == 0 else 4.5,
                'Yaw': 45.0 if self.role == 0 else 225.0,
                'Name': self.role_name,
                'entities': []
            }
            self.prev_info = default_info
            return self._extract_state(default_info), {}
    
    def step(self, action):
        """Step CORREGIDO con caché de última observación válida"""
        try:
            obs, malmo_reward, done, info = self.env.step(action)
        
            # Parsear info
            info_dict = self._parse_info(info)

            # ================================
            # FIX: Validar si info es VÁLIDA
            # ================================
            valid_info = (
                isinstance(info_dict, dict)
                and 'XPos' in info_dict
                and 'ZPos' in info_dict
                and isinstance(info_dict.get('entities', []), list)
                and len(info_dict['entities']) > 0
            )

            if not valid_info:
                # Mantener la última observación válida
                info_dict = self.prev_info
            # Si ES válida, actualizamos prev_info abajo
            # ================================

            # Calcular recompensa personalizada
            custom_reward, captured = self._calculate_reward(info_dict, self.prev_info)
        
            custom_reward = custom_reward if custom_reward is not None else 0.0
            total_reward = custom_reward  # Solo usar custom rewards
        
            # Extraer estado de info
            state = self._extract_state(info_dict)

            # ==== LOGS INTELIGENTES (NO LOS TOCO) ====
            if 'entities' in info_dict:
                enemy = None
                for e in info_dict['entities']:
                    if isinstance(e, dict) and e.get('name') != info_dict.get('Name'):
                        enemy = e
                        break

                if enemy:
                    ex, ez = enemy.get('x', None), enemy.get('z', None)
                    if ex is not None and ez is not None:
                        dx = ex - info_dict.get('XPos', 0)
                        dz = ez - info_dict.get('ZPos', 0)
                        dist = (dx**2 + dz**2)**0.5
                    else:
                        dist = None
                else:
                    dist = None
            else:
                dist = None

            if self.episode_steps % 50 == 0:
                print(f"[{self.role_name}] step={self.episode_steps} reward={total_reward:.3f} dist={dist}")

            # ================================
            # FIX: Solo actualizar prev_info si fue válida
            # ================================
            if valid_info:
                self.prev_info = info_dict
            # ================================

            # Actualizar
            self.episode_steps += 1
            self.episode_reward += total_reward
        
            # Info para callback
            if not isinstance(info, dict):
                info = {}
            info['captured'] = bool(captured)
        
            if captured:
                done = True
            if self.episode_steps >= self.max_steps:
                done = True
                info['TimeLimit.truncated'] = True

            if captured:
                print(f"[{self.role_name}] CAPTURED detected!")

            if done and self.episode_steps < self.max_steps:
                print(f"[{self.role_name}] Episode ended EARLY (captured or env done)")

        
            return (
                np.array(state, dtype=np.float32),
                float(total_reward),
                bool(done),
                False,
                info
            )
    
        except Exception as e:
            print(f"[{self.role_name}] Error en step: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(15, dtype=np.float32), -1.0, True, False, {"captured": False}

    
    def close(self):
        if self.env:
            try:
                self.env.close()
            except:
                pass


# ==================== CALLBACK ====================
class ImprovedCallback(BaseCallback):
    """Callback con logging"""
    
    def __init__(self, role, role_name, check_freq=2000, verbose=1):
        super().__init__(verbose)
        self.role = role
        self.role_name = role_name
        self.check_freq = check_freq
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.max_no_improvement = 200
    
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
        
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards)
            mean_length = np.mean(self.episode_lengths)
            std_reward = np.std(self.episode_rewards)
            
            print(f"\n[{self.role_name}] Step {self.n_calls}")
            print(f"  Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"  Length: {mean_length:.1f}")
            
            if mean_reward > self.best_mean_reward:
                improvement = mean_reward - self.best_mean_reward
                self.best_mean_reward = mean_reward
                self.no_improvement_count = 0
                
                model_path = f"models/{self.role_name.lower()}_best.zip"
                self.model.save(model_path)
                print(f"  ✓ MEJOR (+{improvement:.2f})")
            else:
                self.no_improvement_count += 1
                print(f"  Sin mejora ({self.no_improvement_count}/{self.max_no_improvement})")
            
            if self.no_improvement_count >= self.max_no_improvement:
                print(f"\n[{self.role_name}] ⚠️ EARLY STOPPING")
                return False
        
        return True


# ==================== ENTRENAMIENTO ====================
def train_agent_sb3(role, xml, port, server, server2, total_timesteps, start_barrier):
    """Entrenamiento"""
    role_name = "Perseguidor_PPO" if role == 0 else "Escapista_DQN"
    
    print(f"\n[{role_name}] Inicializando...")
    
    start_barrier.wait()
    time.sleep(role * 2)
    
    model_path = f"models/{role_name}_best.zip"
    
    try:
        env = MalmoGymWrapper(
            xml=xml, port=port, server=server,
            server2=server2, port2=port + role,
            role=role, exp_uid='sb3_fixed_training'
        )
        
        env = Monitor(env, filename=f"logs/{role_name}")
        
        if os.path.exists(model_path):
            print(f"[{role_name}] Cargando modelo: {model_path}")
            if role == 0:
                model = PPO.load(model_path, env=env)
            else:
                model = DQN.load(model_path, env=env)
        else:
            if role == 0:
                print(f"[{role_name}] Creando PPO...")
                model = PPO(
                    "MlpPolicy", env,
                    learning_rate=5e-4, n_steps=2048, batch_size=128,
                    n_epochs=10, gamma=0.995, gae_lambda=0.98,
                    clip_range=0.2, ent_coef=0.02, vf_coef=0.5,
                    max_grad_norm=0.5, verbose=1,
                    tensorboard_log=f"./tensorboard/{role_name}/"
                )
            else:
                print(f"[{role_name}] Creando DQN...")
                model = DQN(
                    "MlpPolicy", env,
                    learning_rate=5e-4, buffer_size=150000,
                    learning_starts=500, batch_size=64, tau=1.0,
                    gamma=0.995, train_freq=4, gradient_steps=2,
                    target_update_interval=500,
                    exploration_fraction=0.4,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.02, verbose=1,
                    tensorboard_log=f"./tensorboard/{role_name}/"
                )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=20000, save_path=f"./models/{role_name}/",
            name_prefix=role_name
        )
        
        custom_callback = ImprovedCallback(
            role=role, role_name=role_name, check_freq=2000
        )
        
        print(f"\n[{role_name}] 🚀 Entrenamiento iniciado ({total_timesteps:,} steps)")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, custom_callback],
            progress_bar=False  # Desactivado para multi-agente
        )
        
        final_path = f"models/{role_name}_FINAL.zip"
        model.save(final_path)
        print(f"\n[{role_name}] ✓ Modelo final: {final_path}")
        
        env.close()
        print(f"[{role_name}] ✓ Completado!")
    
    except Exception as e:
        print(f"[{role_name}] ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


# ==================== MAIN ====================
if __name__ == '__main__':
    print("=" * 70)
    print("  ENTRENAMIENTO MULTI-AGENTE CORREGIDO")
    print("  FIX: Observaciones desde info, no obs")
    print("=" * 70)
    
    Path('models').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    Path('tensorboard').mkdir(exist_ok=True)
    
    xml_path = Path('missions/chase_escape.xml')
    if not xml_path.exists():
        print(f"\n❌ ERROR: No se encuentra {xml_path}")
        exit(1)
    
    xml = xml_path.read_text()
    mission = etree.fromstring(xml)
    number_of_agents = len(mission.findall('{http://ProjectMalmo.microsoft.com}AgentSection'))
    
    if number_of_agents != 2:
        print(f"\n❌ ERROR: Se necesitan 2 agentes")
        exit(1)
    
    print(f"\n✓ Misión cargada: {number_of_agents} agentes")
    
    PORT = 9000
    SERVER = '127.0.0.1'
    SERVER2 = SERVER
    TOTAL_TIMESTEPS = 100000  
    
    print(f"\nConfiguración:")
    print(f"  Puerto base: {PORT}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  🔧 FIX APLICADO: Extrayendo de info en lugar de obs")
    
    print("\n" + "=" * 70)
    print("INSTRUCCIONES:")
    print("  1. DOS terminales:")
    print(f"  2. py -3.7 -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft({PORT})\"")
    print(f"  3. py -3.7 -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft({PORT + 1})\"")
    print("=" * 70)
    print("\nPresiona ENTER cuando estén listas...")
    input()
    
    print("\n🚀 Iniciando en 3 segundos...")
    time.sleep(3)
    
    start_barrier = Barrier(number_of_agents)
    
    threads = [
        Thread(
            target=train_agent_sb3,
            args=(i, xml, PORT, SERVER, SERVER2, TOTAL_TIMESTEPS, start_barrier),
            name=f"Agent-{i}"
        )
        for i in range(number_of_agents)
    ]
    
    [t.start() for t in threads]
    [t.join() for t in threads]
    
    print("\n" + "=" * 70)
    print("  ✓ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print("\nTensorBoard: tensorboard --logdir ./tensorboard")
    print("\n✅ ¡Ahora SÍ debería funcionar!")