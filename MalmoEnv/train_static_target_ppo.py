"""
Entrenamiento PPO para un solo agente que busca un objetivo estático (ArmorStand)
en el mismo mapa de chase_escape: arena 11x11 con obstáculos.
El objetivo se coloca en una posición aleatoria antes de iniciar Malmo.
"""
import malmoenv
import numpy as np
from pathlib import Path
from lxml import etree
import time
import gymnasium as gym
from gymnasium import spaces
import os
import random

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
    print("V Stable-Baselines3 importado correctamente")
except ImportError:
    print("? ERROR: Instala stable-baselines3:")
    print("   pip install stable-baselines3[extra]")
    raise


class TargetEnv(gym.Env):
    """Entorno Gym para buscar un objetivo estático."""
    def __init__(self, xml, port, server, exp_uid):
        super().__init__()
        self.xml = xml
        self.port = port
        self.server = server
        self.exp_uid = exp_uid
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.env = None
        self.prev_obs = {}
        self.episode_steps = 0
        self.max_steps = 500
        self._init_malmo()

    def _init_malmo(self):
        try:
            self.env = malmoenv.make()
            self.env.init(
                self.xml,
                self.port,
                server=self.server,
                role=0,
                exp_uid=self.exp_uid
            )
            print(f"[Buscador] V Conectado al puerto {self.port}")
        except Exception as e:
            print(f"[Buscador] ? Error de conexión: {e}")
            raise
#UN cmabio para subir esto wtf?
    def _extract_state(self, obs):
        """Vectoriza posición propia, yaw y objetivo."""
        features = []
        if not isinstance(obs, dict):
            return np.zeros(10, dtype=np.float32)

        my_x = obs.get('XPos', 0.0)
        my_z = obs.get('ZPos', 0.0)
        my_vx = obs.get('XVel', 0.0)
        my_vz = obs.get('ZVel', 0.0)
        yaw = obs.get('Yaw', 0.0) / 180.0
        features.extend([my_x / 10.0, my_z / 10.0, my_vx, my_vz, yaw])

        target_found = False
        entities = obs.get('entities', [])
        if isinstance(entities, list):
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                name = str(ent.get("name", "")).lower()
                if name.startswith("armorstand"):
                    tx = ent.get('x', my_x)
                    tz = ent.get('z', my_z)
                    rel_x = (tx - my_x) / 20.0
                    rel_z = (tz - my_z) / 20.0
                    distance = np.sqrt(rel_x**2 + rel_z**2)
                    angle = np.arctan2(rel_z, rel_x) / np.pi
                    features.extend([rel_x, rel_z, distance, angle, 0.0])
                    target_found = True
                    break
        if not target_found:
            features.extend([0.0, 0.0, 2.0, 0.0, 0.0])

        features = features[:10]
        while len(features) < 10:
            features.append(0.0)
        return np.array(features, dtype=np.float32)

    def _calculate_reward(self, obs, prev_obs):
        reward = 0.0
        captured = False

        if not isinstance(obs, dict) or not isinstance(prev_obs, dict):
            return reward, captured

        my_x = obs.get('XPos', 0.0)
        my_z = obs.get('ZPos', 0.0)
        prev_x = prev_obs.get('XPos', my_x)
        prev_z = prev_obs.get('ZPos', my_z)

        movement_penalty = -0.01

        target_dist = None
        prev_target_dist = None
        entities = obs.get('entities', [])
        if isinstance(entities, list):
            for ent in entities:
                if isinstance(ent, dict):
                    name = str(ent.get("name", "")).lower()
                    if name.startswith("armorstand"):
                        tx = ent.get('x', my_x)
                        tz = ent.get('z', my_z)
                        target_dist = np.sqrt((my_x - tx) ** 2 + (my_z - tz) ** 2)
                        break

        prev_entities = prev_obs.get('entities', [])
        if isinstance(prev_entities, list):
            for ent in prev_entities:
                if isinstance(ent, dict):
                    name = str(ent.get("name", "")).lower()
                    if name.startswith("armorstand"):
                        tx = ent.get('x', prev_x)
                        tz = ent.get('z', prev_z)
                        prev_target_dist = np.sqrt((prev_x - tx) ** 2 + (prev_z - tz) ** 2)
                        break

        if target_dist is not None:
            if prev_target_dist is not None:
                delta = prev_target_dist - target_dist
                reward += delta * 2.0

            if target_dist < 1.0:
                reward += 100.0
                captured = True
            elif target_dist < 2.0:
                reward += 5.0
            elif target_dist < 3.5:
                reward += 1.0
            else:
                reward -= 0.5

        if abs(my_x) > 9.5 or abs(my_z) > 9.5:
            reward -= 10.0

        return reward + movement_penalty, captured

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            obs = self.env.reset()
            self.prev_obs = obs if isinstance(obs, dict) else {}
            self.episode_steps = 0
            state = self._extract_state(obs)
            return state, {}
        except Exception as e:
            print(f"[Buscador] Error en reset: {e}")
            return np.zeros(10, dtype=np.float32), {}

    def step(self, action):
        try:
            obs, malmo_reward, done, info = self.env.step(action)
            if not isinstance(info, dict):
                info = {"raw_info": info}

            custom_reward, captured = self._calculate_reward(obs, self.prev_obs)

            malmo_reward = malmo_reward if malmo_reward is not None else 0.0
            custom_reward = custom_reward if custom_reward is not None else 0.0
            total_reward = malmo_reward + custom_reward

            state = self._extract_state(obs)
            self.prev_obs = obs if isinstance(obs, dict) else {}
            self.episode_steps += 1

            info["captured"] = bool(captured)
            if captured:
                done = True
            if self.episode_steps >= self.max_steps:
                done = True
                info["TimeLimit.truncated"] = True

            return np.array(state, dtype=np.float32), float(total_reward), bool(done), False, info
        except Exception as e:
            print(f"[Buscador] Error en step: {e}")
            return np.zeros(10, dtype=np.float32), -1.0, True, False, {"captured": False}

    def close(self):
        if self.env:
            try:
                self.env.close()
            except Exception:
                pass


class RewardMonitorCallback(BaseCallback):
    """Imprimir progreso y guardar mejor modelo."""
    def __init__(self, check_freq=2000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0 and len(self.model.ep_info_buffer) > 0:
            rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            mean_r = np.mean(rewards)
            print(f"[Buscador] Paso {self.n_calls} | Reward prom (ep buffer): {mean_r:.2f}")
            if mean_r > self.best_mean:
                self.best_mean = mean_r
                path = "models/Buscador_PPO/buscador_best.zip"
                Path("models/Buscador_PPO").mkdir(parents=True, exist_ok=True)
                self.model.save(path)
                print(f"[Buscador] V Nuevo mejor modelo guardado en {path}")
        return True


def place_random_target(xml_text):
    """Coloca el ArmorStand en x/z aleatorios dentro de la arena."""
    mission = etree.fromstring(xml_text)
    ns = {"m": "http://ProjectMalmo.microsoft.com"}
    target = mission.find(".//m:DrawEntity[@type='ArmorStand']", namespaces=ns)
    if target is not None:
        target.set("x", str(random.uniform(-4.0, 4.0)))
        target.set("z", str(random.uniform(-4.0, 4.0)))
    return etree.tostring(mission).decode("utf-8")


if __name__ == "__main__":
    print("=" * 70)
    print("  ENTRENAMIENTO PPO: BUSCAR OBJETIVO ESTÁTICO")
    print("=" * 70)

    Path("models/Buscador_PPO").mkdir(parents=True, exist_ok=True)
    Path("logs/Buscador_PPO").mkdir(parents=True, exist_ok=True)
    Path("tensorboard/Buscador_PPO").mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    xml_path = script_dir / "missions" / "chase_static_target.xml"
    if not xml_path.exists():
        print(f"? ERROR: No se encuentra {xml_path}")
        raise SystemExit(1)

    xml_base = xml_path.read_text()
    # Posición aleatoria del objetivo antes de iniciar Malmo
    xml = place_random_target(xml_base)

    mission = etree.fromstring(xml)
    num_agents = len(mission.findall('{http://ProjectMalmo.microsoft.com}AgentSection'))
    if num_agents != 1:
        print(f"? ERROR: La misión debe tener 1 agente (encontrados: {num_agents})")
        raise SystemExit(1)

    PORT = 9000
    SERVER = "127.0.0.1"
    TOTAL_TIMESTEPS = 100_000

    print(f"Puerto: {PORT}")
    print(f"Timesteps: {TOTAL_TIMESTEPS:,}")
    print("\nInstrucciones:")
    print(f"  1) Abre una terminal y lanza Minecraft: py -3.7 -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft({PORT})\"")
    print("  2) Espera a ver 'SERVER STARTED'")
    print("  3) Vuelve aquí y presiona ENTER para iniciar el entrenamiento")
    input()

    try:
        env = TargetEnv(
            xml=xml,
            port=PORT,
            server=SERVER,
            exp_uid="ppo_static_target"
        )
        env = Monitor(env, filename="logs/Buscador_PPO")

        model_path = "models/Buscador_PPO/buscador_best.zip"
        if os.path.exists(model_path):
            print(f"[Buscador] Cargando modelo existente: {model_path}")
            model = PPO.load(model_path, env=env)
        else:
            print("[Buscador] Creando modelo PPO...")
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
                tensorboard_log="./tensorboard/Buscador_PPO/"
            )

        checkpoint = CheckpointCallback(
            save_freq=10_000,
            save_path="./models/Buscador_PPO/",
            name_prefix="Buscador_PPO"
        )
        reward_cb = RewardMonitorCallback(check_freq=2_000)

        print("\n[Buscador] Iniciando entrenamiento...")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint, reward_cb],
            progress_bar=False
        )

        final_path = "models/Buscador_PPO_FINAL.zip"
        model.save(final_path)
        print(f"\n[Buscador] V Modelo final guardado: {final_path}")
        env.close()
        print("[Buscador] V Entrenamiento completado.")
    except Exception as e:
        print(f"[Buscador] ? ERROR: {e}")
        import traceback
        traceback.print_exc()
