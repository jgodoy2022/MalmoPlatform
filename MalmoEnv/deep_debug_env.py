"""
DEBUG para la mision chase_static_target (ArmorStand estatico).
- Inyecta el target en coordenadas aleatorias
- Valida la conexion al servidor Malmo y muestra lo que llega en obs/info
- Corre pasos aleatorios para inspeccionar recompensas y flags de finalizacion

Uso rapido:
1) Lanza Minecraft: py -3.7 -c "import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9000)" y espera SERVER STARTED
2) Ejecuta: py -3.7 deep_debug_env.py
3) Sigue la consola; imprime obs/info detallados y detiene al terminar la mision o completar los steps configurados (por defecto 30)

DEBUG para chase_static_target (ArmorStand estático).
Imprime TODO lo que recibe el agente y valida que el target esté presente.
"""
import malmoenv
import socket
import time
import numpy as np
from pathlib import Path
from lxml import etree
import random
import sys

# --------- UTIL helpers ----------
def wait_for_server(host, port, timeout=60):
    """Espera a que el socket host:port responda (TCP) hasta timeout; devuelve True si conecta."""
    print(f"Esperando servidor en {host}:{port} ...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            res = s.connect_ex((host, port))
            s.close()
            if res == 0:
                print(f"Servidor {host}:{port} listo.")
                return True
        except Exception:
            pass
        time.sleep(1.0)
    print(f"TIMEOUT esperando servidor {host}:{port}")
    return False

def place_random_target_in_xml(xml_text):
    """Coloca ArmorStand en coordenadas aleatorias dentro de arena (si existe DrawEntity ArmorStand)."""
    mission = etree.fromstring(xml_text)
    ns = {"m": "http://ProjectMalmo.microsoft.com"}
    target = mission.find(".//m:DrawEntity[@type='ArmorStand']", namespaces=ns)
    if target is not None:
        x = round(random.uniform(-4.0, 4.0), 3)
        z = round(random.uniform(-4.0, 4.0), 3)
        target.set("x", str(x))
        target.set("z", str(z))
        print(f"ArmorStand colocado en x={x}, z={z}")
    else:
        print("No se encontró DrawEntity type='ArmorStand' en el XML.")
    return etree.tostring(mission).decode("utf-8")

# --------- DEBUG main ----------
def debug_mission(xml_text, port=9000, server="127.0.0.1", exp_uid="debug_static_target", steps=30):
    """Inicializa Malmo, hace reset y ejecuta 'steps' acciones aleatorias para inspeccionar obs/info y recompensas."""
    print("Inicializando env de debug...")
    env = None
    try:
        env = malmoenv.make()
        # IMPORTANT: use role=0 (single-agent mission)
        env.init(xml_text, port, server=server, role=0, exp_uid=exp_uid)
        print("env.init() OK")
    except Exception as e:
        print("ERROR en env.init():", e)
        if env:
            try:
                env.close()
            except:
                pass
        raise

    try:
        print("-> reset()")
        obs = env.reset()
        print_obs_info(obs, None, prefix="[RESET]")

        for i in range(steps):
            action = env.action_space.sample()
            # acción aleatoria para test. Ajusta si quieres mapping específico.
            print(f"\nSTEP {i+1}/{steps} -> Acción sampleada: {action}")
            obs, reward, done, info = env.step(action)
            print_obs_info(obs, info, prefix=f"[STEP {i+1}]")
            print(f" Reward raw: {reward} | Done: {done} | Info-type: {type(info)}")
            if done:
                print("Misión terminó (done=True). Salimos del bucle de debug.")
                break
            time.sleep(0.8)
    except Exception as e:
        print("ERROR durante run:", e)
        import traceback; traceback.print_exc()
    finally:
        if env:
            try:
                env.close()
                print("Entorno cerrado correctamente.")
            except Exception as e:
                print("Error cerrando entorno:", e)

def print_obs_info(obs, info, prefix=""):
    """Imprime tipo y claves de obs/info, posiciones y primeras entidades detectadas para depurar la mision."""
    print("\n" + "-"*80)
    print(f"{prefix} OBS type: {type(obs)}")
    if isinstance(obs, dict):
        keys = list(obs.keys())
        print(f"{prefix} Claves OBS: {keys}")
        # posiciones propias
        print(f"{prefix} Pos propia: XPos={obs.get('XPos','N/A')}  ZPos={obs.get('ZPos','N/A')}  Yaw={obs.get('Yaw','N/A')}")
        # entidades
        entities = obs.get('entities', [])
        print(f"{prefix} ENTIDADES: tipo {type(entities)} | count={len(entities) if isinstance(entities, list) else 'N/A'}")
        if isinstance(entities, list) and len(entities) > 0:
            for idx, ent in enumerate(entities[:10]):
                if isinstance(ent, dict):
                    print(f"  - Ent[{idx}]: name={ent.get('name')} type={ent.get('type')} pos=({ent.get('x')},{ent.get('y')},{ent.get('z')}) yaw={ent.get('yaw')}")
        else:
            print(f"{prefix} (No entities detected in obs)")
        # board/grid if present
        if 'board' in obs:
            b = obs.get('board')
            if isinstance(b, list):
                print(f"{prefix} BOARD len={len(b)} first10={b[:10]}")
    else:
        print(f"{prefix} OBS NO es dict. Contenido (trunc): {str(obs)[:300]}")
    # info
    print(f"{prefix} INFO type: {type(info)}")
    if isinstance(info, dict):
        print(f"{prefix} INFO keys: {list(info.keys())}")
        # print some likely keys
        for k in ['caught_the_Chicken','Agent0_defaulted','Agent1_defaulted','TimeLimit.truncated']:
            if k in info:
                print(f"  INFO[{k}] = {info[k]}")
    else:
        print(f"{prefix} INFO is not dict or empty: {info}")
    print("-"*80 + "\n")

# --------- ENTRYPOINT ----------
if __name__ == "__main__":
    random.seed(1234)
    script_dir = Path(__file__).resolve().parent
    xml_path = script_dir / "missions" / "chase_static_target.xml"
    if not xml_path.exists():
        print("ERROR: no se encontró missions/chase_static_target.xml")
        sys.exit(1)

    xml_base = xml_path.read_text()
    xml_with_target = place_random_target_in_xml(xml_base)

    PORT = 9000
    SERVER = "127.0.0.1"
    print("INSTRUCCIONES:")
    print(f"  1) Abre una terminal y lanza Minecraft: py -3.7 -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft({PORT})\"")
    print("  2) Espera a que en la ventana de Minecraft salga: SERVER STARTED")
    print("  3) Vuelve aquí y presiona ENTER para continuar")
    input("Presiona ENTER cuando Minecraft muestre SERVER STARTED...")

    # comprobar servidor
    if not wait_for_server(SERVER, PORT, timeout=40):
        print("ERROR: servidor no responde en el puerto", PORT)
        sys.exit(1)

    print("Lanzando debug de misión (30 steps)...")
    debug_mission(xml_with_target, port=PORT, server=SERVER, exp_uid="debug_static_target_001", steps=30)
