"""
Entrenamiento single-agent PPO para perseguir un objetivo estatico (ArmorStand) en Malmo.

Uso rapido (Windows):
1) Levanta Minecraft en el puerto deseado: py -3.7 -c "import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9000)"
2) Desde MalmoEnv: py -3.7 train_static_target_ppo.py
   (continua desde el modelo guardado si existe en models/)

Detalles:
- Observacion vectorial de 10 dims, accion discreta de 4 movimientos
- Usa Stable-Baselines3.PPO con callbacks de checkpoint
- Maximo de pasos por episodio fijado en 450; recompensas dependen de distancia al target
"""

import malmoenv
import numpy as np
from pathlib import Path
from lxml import etree
import time
import gymnasium as gym
from gymnasium import spaces
import json
import os

# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor


class TargetEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, xml, port, server, exp_uid, fixed_target=None):#(0.0, 0.0)):
        super().__init__()
        self.xml = xml
        self.port = port
        self.server = server
        self.exp_uid = exp_uid

        # Observación 10-dim (compatible con tu anterior)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.env = None
        self.prev_info = {}        # cache para última info válida
        self.episode_steps = 0
        self.max_steps = 450

        # fixed_target: si NO es None, usamos coordenadas conocidas (ayuda al agente)
        # formato: (tx, tz)
        self.fixed_target = fixed_target

        self._init_malmo()

    def _init_malmo(self):
        print(f"[Env] Iniciando Malmo en puerto {self.port}...")
        self.env = malmoenv.make()
        # NO existe set_available_actions() en malmoenv.Env -> no llamarlo
        self.env.init(self.xml, self.port, server=self.server, role=0, exp_uid=self.exp_uid)
        print(f"[Env] Conectado a {self.server}:{self.port}")

    def _parse_info(self, info):
        """Parsea 'info' que puede venir como dict, json string, o cosas raras."""
        if info is None:
            return {}
        if isinstance(info, dict):
            return info
        if isinstance(info, str):
            try:
                return json.loads(info)
            except Exception:
                try:
                    return eval(info)
                except Exception:
                    return {}
        return {}

    def _get_target_position_from_info(self, info):
        """Intenta leer ArmorStand desde info['entities']. devuelve (tx, tz) o (None, None)."""
        if not isinstance(info, dict):
            return None, None
        for ent in info.get("entities", []):
            if not isinstance(ent, dict):
                continue
            ttype = str(ent.get("type", "")).lower()
            name = str(ent.get("name", "")).lower()
            if ttype == "armorstand" or "armor" in name or "armorstand" in name:
                return ent.get("x", None), ent.get("z", None)
        return None, None

    def _extract_state(self, info):
        """Construye vector 10D desde info (fallbacks si faltan datos)."""
        if not isinstance(info, dict):
            return np.zeros(10, dtype=np.float32)

        X = info.get("XPos", 0.0)
        Z = info.get("ZPos", 0.0)
        vx = info.get("motionX", 0.0)
        vz = info.get("motionZ", 0.0)
        yaw = info.get("Yaw", 0.0) / 180.0

        features = [X/10.0, Z/10.0, vx, vz, yaw]

        # preferir fixed_target si está definido
        if self.fixed_target is not None:
            tx, tz = self.fixed_target
            rel_x = np.clip((tx - X) / 20.0, -1, 1)
            rel_z = np.clip((tz - Z) / 20.0, -1, 1)
            dist = np.sqrt((tx - X)**2 + (tz - Z)**2) / 10.0
            dist = np.clip(dist, 0, 2)
            angle = np.arctan2(tz - Z, tx - X) / np.pi
            features.extend([rel_x, rel_z, dist, angle, 0.0])
            return np.array(features[:10], dtype=np.float32)

        # si no hay fixed target, intentar extraer de entities
        target_features = [0.0, 0.0, 2.0, 0.0, 0.0]
        for ent in info.get("entities", []):
            if isinstance(ent, dict):
                ttype = str(ent.get("type", "")).lower()
                name = str(ent.get("name", "")).lower()
                if ttype == "armorstand" or "armor" in name:
                    tx = ent.get("x", X)
                    tz = ent.get("z", Z)
                    rel_x = np.clip((tx - X) / 20.0, -1, 1)
                    rel_z = np.clip((tz - Z) / 20.0, -1, 1)
                    dist = np.sqrt((tx - X)**2 + (tz - Z)**2) / 10.0
                    dist = np.clip(dist, 0, 2)
                    angle = np.arctan2(tz - Z, tx - X) / np.pi
                    target_features = [rel_x, rel_z, dist, angle, 0.0]
                    break

        features.extend(target_features)
        # asegurar longitud
        features = features[:10]
        while len(features) < 10:
            features.append(0.0)
        return np.array(features, dtype=np.float32)

    def _calculate_reward(self, info, prev_info):
        """Reward más estable, menos hackeable y enfocado en tocar el target."""
        
        if not isinstance(info, dict):
            return -0.02, False  # castigo leve por no tener info

        X = info.get("XPos", 0.0)
        Z = info.get("ZPos", 0.0)
        captured = False
        reward = 0.0

        # === Movimiento (penaliza girar sin avanzar) ===
        # penalización base por paso
        reward -= 0.01  

        # Detectar si se movió
        moved = False
        if isinstance(prev_info, dict):
            prev_X = prev_info.get("XPos", X)
            prev_Z = prev_info.get("ZPos", Z)
            dist_moved = np.sqrt((X - prev_X)**2 + (Z - prev_Z)**2)
            moved = dist_moved > 0.01

        if not moved:
            reward -= 0.015  # castigo por quedarse quieto (anti-giro infinito)


        # === DISTANCIA AL TARGET (igual que antes) ===
        if self.fixed_target is not None:
            tx, tz = self.fixed_target
            dist = np.sqrt((X - tx)**2 + (Z - tz)**2)

            if isinstance(prev_info, dict):
                prev_dist = np.sqrt((prev_info.get("XPos", X) - tx)**2 +
                                    (prev_info.get("ZPos", Z) - tz)**2)
            else:
                prev_dist = None
        else:
            # buscar ArmorStand (igual que antes)
            def get_dist(dic):
                if not isinstance(dic, dict):
                    return None
                for ent in dic.get("entities", []):
                    if isinstance(ent, dict) and str(ent.get("type", "")).lower() == "armorstand":
                        return np.sqrt((dic.get("XPos",0.0) - ent.get("x",0.0))**2 +
                                    (dic.get("ZPos",0.0) - ent.get("z",0.0))**2)
                return None
            dist = get_dist(info)
            prev_dist = get_dist(prev_info) if isinstance(prev_info, dict) else None


        # === Recompensa por progreso ===
        if dist is not None and prev_dist is not None:
            delta = prev_dist - dist  # positivo si mejora

            reward += delta * 3.0  # más fuerte para incentivar acercarse

            # si se está alejando consistentemente → penalización
            if delta < 0:
                reward -= 0.02


        # === Recompensas por cercanía (rebalanceado) ===
        if dist is not None:
            if dist < 0.8:
                reward += 50.0
                captured = True
            elif dist < 1.5:
                reward += 3.0
            elif dist < 3.0:
                reward += 0.5
            else:
                reward -= 0.2


        # === penalización por salir del área ===
        if abs(X) > 9.3 or abs(Z) > 9.3:
            reward -= 20.0


        return float(reward), bool(captured)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0

        # Reset del entorno Malmo (puede devolver tuplas varias)
        raw = self.env.reset()

        # intentamos extraer la "info" (último elemento con datos)
        parsed_info = {}
        if isinstance(raw, tuple):
            for v in raw[::-1]:
                p = self._parse_info(v)
                if p:
                    parsed_info = p
                    break
        else:
            parsed_info = self._parse_info(raw)

        # Si no hay info válida, esperar un poco y hacer un step dummy
        if not isinstance(parsed_info, dict) or "XPos" not in parsed_info:
            # hacer small wait + dummy step para forzar que Malmo reporte datos válidos
            time.sleep(1.0)
            try:
                out = self.env.step(0)  # acción 0 (adelante) como dummy
                # parsear info del dummy
                if isinstance(out, tuple):
                    # preferir último elemento
                    candidate = out[-1]
                else:
                    candidate = out
                parsed_info = self._parse_info(candidate) or parsed_info
            except Exception:
                pass

        # si aun así no existe XPos, usar defaults (posición inicial del XML)
        if not isinstance(parsed_info, dict) or "XPos" not in parsed_info:
            parsed_info = {"XPos": -4.5, "ZPos": -4.5, "Yaw": 45.0, "entities": []}

        self.prev_info = parsed_info.copy() if isinstance(parsed_info, dict) else {}
        state = self._extract_state(parsed_info)
        return state, {}

    def step(self, action):
        try:
            out = self.env.step(action)

            # Normalizar formatos: malmoenv puede devolver (obs, rew, done, info) o (obs, info)
            obs_part = None
            malmo_rew = 0.0
            done_flag = False
            raw_info = {}

            if isinstance(out, tuple):
                if len(out) == 4:
                    obs_part, malmo_rew, done_flag, raw_info = out
                elif len(out) == 2:
                    obs_part, raw_info = out
                else:
                    raw_info = out[-1] if len(out) > 0 else {}
            else:
                raw_info = out

            info = self._parse_info(raw_info)

            # Si info no es válido, usamos prev_info (cache)
            if not (isinstance(info, dict) and "XPos" in info and "ZPos" in info):
                info = self.prev_info

            # calcular recompensa y captura
            custom_reward, captured = self._calculate_reward(info, self.prev_info or {})

            # extraer estado
            state = self._extract_state(info)

            # actualizar cache si la info fue válida
            if isinstance(info, dict):
                self.prev_info = info.copy()

            self.episode_steps += 1
            terminated = bool(captured)
            truncated = self.episode_steps >= self.max_steps

            # preparar info de salida para SB3/Monitor
            out_info = {"captured": bool(captured)}
            if terminated or truncated:
                # resúmen de episodio para que Monitor registre
                out_info["episode"] = {
                    "r": float(custom_reward),
                    "l": int(self.episode_steps)
                }

            return np.array(state, dtype=np.float32), float(custom_reward), terminated, truncated, out_info

        except Exception as e:
            print(f"[Env] Error en step: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(self.observation_space.shape, dtype=np.float32), -1.0, True, False, {"captured": False}

    def close(self):
        try:
            if self.env:
                self.env.close()
        except Exception:
            pass


class RewardMonitorCallback(BaseCallback):
    def __init__(self, check_freq=2000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean = -np.inf

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                print(f"[Callback] step={self.n_calls} | mean={mean_r:.2f}")
                if mean_r > self.best_mean:
                    self.best_mean = mean_r
                    Path("models/Buscador_PPO").mkdir(exist_ok=True)
                    self.model.save("models/Buscador_PPO/best_model.zip")
                    print("[Callback] Guardado nuevo mejor modelo")
        return True


# -------------------- MAIN --------------------
if __name__ == "__main__":
    print("="*70)
    print("     ENTRENAMIENTO PPO — TARGET ESTÁTICO (0,0)")
    print("="*70)

    Path("models/Buscador_PPO").mkdir(parents=True, exist_ok=True)
    Path("logs/Buscador_PPO").mkdir(parents=True, exist_ok=True)
    Path("tensorboard/Buscador_PPO").mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    xml_path = script_dir / "missions" / "chase_static_target.xml"
    if not xml_path.exists():
        print(f"ERROR: No se encuentra {xml_path}")
        raise SystemExit(1)

    xml = xml_path.read_text()

    PORT = 9000
    SERVER = "127.0.0.1"
    TIMESTEPS = 250_000

    input("\nLanza Minecraft (malmoenv.bootstrap.launch_minecraft(PORT)) y presiona ENTER para entrenar...")

    # fijamos el target en (0,0) para ayudar el aprendizaje inicial
    FIXED_TARGET = (0.0, 0.0)

    env = TargetEnv(xml=xml, port=PORT, server=SERVER, exp_uid="ppo_target_static", fixed_target=FIXED_TARGET)
    env = Monitor(env, "logs/Buscador_PPO")

    model_path = "models/Buscador_PPO/best_model.zip"
    if os.path.exists(model_path):
        print("[INFO] Cargando modelo previo...")
        model = PPO.load(model_path, env=env)
    else:
        print("[INFO] Creando modelo PPO...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            gamma=0.97,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="tensorboard/Buscador_PPO"
        )

    checkpoint = CheckpointCallback(save_freq=10_000, save_path="./models/Buscador_PPO", name_prefix="checkpoint")
    best_reward_callback = RewardMonitorCallback(check_freq=2000)

    model.learn(total_timesteps=TIMESTEPS, callback=[checkpoint, best_reward_callback])

    model.save("models/Buscador_PPO/final_model.zip")
    print("\nEntrenamiento completado")
