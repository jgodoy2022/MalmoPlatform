"""
SCRIPT DE DEBUGGING PARA MALMO
Verifica qué observa realmente cada agente y si detecta al otro

EJECUTAR ESTO PRIMERO antes de entrenar para validar observaciones
"""
import malmoenv
import numpy as np
from pathlib import Path
from lxml import etree
import json
import time
from threading import Thread, Barrier

# ==================== DEBUGGING WRAPPER ====================
class DebugMalmoEnv:
    """Wrapper simple para debuggear observaciones"""
    
    def __init__(self, xml, port, server, server2, port2, role, exp_uid):
        self.role = role
        self.role_name = "Perseguidor" if role == 0 else "Escapista"
        self.xml = xml
        self.port = port
        self.server = server
        self.server2 = server2
        self.port2 = port2
        self.exp_uid = exp_uid
        
        self.env = None
        self.step_count = 0
        
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
            print(f"[{self.role_name}] ✓ Conectado al puerto {self.port if self.role == 0 else self.port2}")
        except Exception as e:
            print(f"[{self.role_name}] ✗ Error: {e}")
            raise
    
    def print_observation_details(self, obs):
        """Imprime TODOS los detalles de la observación"""
        print("\n" + "="*80)
        print(f"[{self.role_name}] STEP {self.step_count} - OBSERVACIÓN COMPLETA")
        print("="*80)
        
        # Tipo de observación
        print(f"\n📦 Tipo de observación: {type(obs)}")
        
        if isinstance(obs, dict):
            print(f"\n📋 Claves disponibles: {list(obs.keys())}")
            
            # Posición propia
            print("\n🤖 MI POSICIÓN:")
            print(f"  X: {obs.get('XPos', 'N/A')}")
            print(f"  Y: {obs.get('YPos', 'N/A')}")
            print(f"  Z: {obs.get('ZPos', 'N/A')}")
            print(f"  Yaw: {obs.get('Yaw', 'N/A')}")
            print(f"  Pitch: {obs.get('Pitch', 'N/A')}")
            
            # Velocidad
            print("\n🏃 MI VELOCIDAD:")
            print(f"  VX: {obs.get('XVel', 'N/A')}")
            print(f"  VY: {obs.get('YVel', 'N/A')}")
            print(f"  VZ: {obs.get('ZVel', 'N/A')}")
            
            # Vida y stats
            print("\n❤️ STATS:")
            print(f"  Life: {obs.get('Life', 'N/A')}")
            print(f"  Food: {obs.get('Food', 'N/A')}")
            
            # CRÍTICO: Entidades
            entities = obs.get('entities', [])
            print(f"\n👥 ENTIDADES DETECTADAS: {len(entities) if isinstance(entities, list) else 0}")
            
            if isinstance(entities, list) and len(entities) > 0:
                for i, entity in enumerate(entities):
                    if isinstance(entity, dict):
                        print(f"\n  Entidad #{i+1}:")
                        print(f"    Nombre: {entity.get('name', 'N/A')}")
                        print(f"    Posición: ({entity.get('x', 'N/A')}, {entity.get('y', 'N/A')}, {entity.get('z', 'N/A')})")
                        print(f"    Yaw: {entity.get('yaw', 'N/A')}")
                        print(f"    Pitch: {entity.get('pitch', 'N/A')}")
                        print(f"    Motion: ({entity.get('motionX', 'N/A')}, {entity.get('motionY', 'N/A')}, {entity.get('motionZ', 'N/A')})")
                        
                        # Calcular distancia si es otro agente
                        name = str(entity.get('name', ''))
                        if 'Agent' in name or 'Perseguidor' in name or 'Escapista' in name:
                            my_x = obs.get('XPos', 0)
                            my_z = obs.get('ZPos', 0)
                            enemy_x = entity.get('x', 0)
                            enemy_z = entity.get('z', 0)
                            dist = np.sqrt((my_x - enemy_x)**2 + (my_z - enemy_z)**2)
                            print(f"    🎯 DISTANCIA AL ENEMIGO: {dist:.2f} bloques")
                            
                            # Verificar captura
                            if dist < 1.5:
                                print(f"    ⚠️ ¡CAPTURA! (dist < 1.5)")
            else:
                print("  ⚠️ ¡NO SE DETECTARON ENTIDADES!")
            
            # Grid/Board
            board = obs.get('board', None)
            if board is not None:
                print(f"\n🗺️ BOARD/GRID:")
                print(f"  Tipo: {type(board)}")
                if isinstance(board, list):
                    print(f"  Tamaño: {len(board)} elementos")
                    # Mostrar algunos elementos
                    print(f"  Primeros 10: {board[:10]}")
            
            # Otros campos relevantes
            other_keys = [k for k in obs.keys() if k not in [
                'XPos', 'YPos', 'ZPos', 'Yaw', 'Pitch',
                'XVel', 'YVel', 'ZVel', 'Life', 'Food',
                'entities', 'board'
            ]]
            if other_keys:
                print(f"\n📊 OTROS CAMPOS:")
                for key in other_keys:
                    print(f"  {key}: {obs.get(key)}")
        
        else:
            print(f"\n⚠️ OBSERVACIÓN NO ES DICCIONARIO")
            print(f"Contenido: {obs}")
        
        print("\n" + "="*80 + "\n")
    
    def run_debug_episode(self, max_steps=20):
        """Ejecuta un episodio corto mostrando todas las observaciones"""
        print(f"\n[{self.role_name}] 🔍 Iniciando episodio de debugging ({max_steps} steps)...\n")
        
        try:
            # Reset
            obs = self.env.reset()
            self.step_count = 0
            self.print_observation_details(obs)
            
            # Ejecutar varios steps
            for step in range(max_steps):
                # Acción aleatoria (0=Norte, 1=Sur, 2=Este, 3=Oeste)
                action = np.random.randint(0, 4)
                action_names = ["Norte", "Sur", "Este", "Oeste"]
                
                print(f"\n[{self.role_name}] 🎮 Ejecutando acción: {action_names[action]}")
                
                obs, reward, done, info = self.env.step(action)
                self.step_count += 1
                
                # Mostrar recompensa de Malmo
                print(f"💰 Reward de Malmo: {reward}")
                print(f"🏁 Done: {done}")
                print(f"ℹ️ Info: {info}")
                
                # Mostrar observación completa
                self.print_observation_details(obs)
                
                if done:
                    print(f"\n[{self.role_name}] ⚠️ Episodio terminado en step {step+1}")
                    break
                
                # Pausa para leer
                time.sleep(1)
            
            print(f"\n[{self.role_name}] ✓ Debug completado")
        
        except Exception as e:
            print(f"\n[{self.role_name}] ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.env:
                try:
                    self.env.close()
                except:
                    pass


# ==================== TEST DE DETECCIÓN DE ENEMIGO ====================
def test_enemy_detection(role, xml, port, server, server2, start_barrier):
    """Test específico para verificar detección del enemigo"""
    role_name = "Perseguidor" if role == 0 else "Escapista"
    
    print(f"\n[{role_name}] Esperando sincronización...")
    start_barrier.wait()
    time.sleep(role * 2)
    
    debug_env = DebugMalmoEnv(
        xml=xml,
        port=port,
        server=server,
        server2=server2,
        port2=port + role,
        role=role,
        exp_uid='debug_test'
    )
    
    debug_env.run_debug_episode(max_steps=20)


# ==================== MAIN ====================
if __name__ == '__main__':
    print("=" * 80)
    print("  🔍 DEBUGGING DE OBSERVACIONES MALMO")
    print("  Verificando qué ve cada agente")
    print("=" * 80)
    
    # Cargar XML
    xml_path = Path('missions/chase_escape.xml')
    if not xml_path.exists():
        print(f"\n❌ ERROR: No se encuentra {xml_path}")
        exit(1)
    
    xml = xml_path.read_text()
    
    # Verificar agentes
    mission = etree.fromstring(xml)
    number_of_agents = len(mission.findall('{http://ProjectMalmo.microsoft.com}AgentSection'))
    
    if number_of_agents != 2:
        print(f"\n❌ ERROR: Se necesitan 2 agentes")
        exit(1)
    
    print(f"\n✓ Misión cargada: {number_of_agents} agentes")
    
    # Configuración
    PORT = 9000
    SERVER = '127.0.0.1'
    SERVER2 = SERVER
    
    print(f"\n⚙️ Configuración:")
    print(f"  Puerto Perseguidor: {PORT}")
    print(f"  Puerto Escapista: {PORT + 1}")
    
    print("\n" + "=" * 80)
    print("INSTRUCCIONES:")
    print("  1. Abre DOS terminales")
    print(f"  2. Terminal 1: py -3.7 -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft({PORT})\"")
    print(f"  3. Terminal 2: py -3.7 -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft({PORT + 1})\"")
    print("  4. Espera 'SERVER STARTED' en ambas")
    print("=" * 80)
    print("\n⚠️ Este script ejecutará 20 steps y mostrará TODO lo que observa cada agente")
    print("Presiona ENTER cuando estés listo...")
    input()
    
    print("\n🚀 Iniciando debug en 3 segundos...")
    time.sleep(3)
    
    # Barrier
    start_barrier = Barrier(number_of_agents)
    
    # Crear threads
    threads = [
        Thread(
            target=test_enemy_detection,
            args=(i, xml, PORT, SERVER, SERVER2, start_barrier),
            name=f"Agent-{i}"
        )
        for i in range(number_of_agents)
    ]
    
    # Iniciar
    [t.start() for t in threads]
    [t.join() for t in threads]
    
    print("\n" + "=" * 80)
    print("  ✓ DEBUG COMPLETADO")
    print("=" * 80)
    print("\n📋 ANÁLISIS:")
    print("  1. ¿Los agentes detectan al otro en 'entities'?")
    print("  2. ¿Las posiciones son correctas?")
    print("  3. ¿La distancia se calcula bien?")
    print("  4. ¿Hay algún campo faltante?")
    print("\n💡 Si NO detecta entidades, el problema está en el XML de la misión")
    print("💡 Si detecta pero posiciones son incorrectas, problema en extracción")
    print("💡 Si todo es None/N/A, Malmo no está enviando observaciones correctamente")