# MalmoEnv (Prototype) #

MalmoEnv is an OpenAI "gym" like Python Environment for Malmo/Minecraft, directly implemented Python to Java Minecraft.

A python "gym env" is created and used to run an agent in a Malmo mission. Each env has a remote Minecraft instance
associated to it (by DNS name or IP and Port). For multi-agent missions, the first agent's (role 0) Minecraft 
client instance is used as a coordinator to allow all agents to rendezvous on mission starts (i.e. on env resets).

As it's python only, you just need this one package, its direct dependencies and (Java) Minecraft!

## Examples of use: ##

Install dependencies:

Java8 JDK, python3, git

`pip install gym lxml numpy pillow`

To prepare Minecraft (after cloning this repository with 
`git clone -b malmoenv https://github.com/Microsoft/malmo.git`):

`cd Minecraft`

`(echo -n "malmo.version=" && cat ../VERSION) > ./src/main/resources/version.properties` 

Running a single agent example mission (run each command in different cmd prompt/shells):

`./launchClient.sh -port 9000 -env` or (On Windows) `launchClient.bat -port 9000 -env`

(In another shell) `cd MalmoEnv` optionally run `python3 setup.py install`

`python3 run.py --mission missions/mobchase_single_agent.xml --port 9000 --episodes 10`

A two agent example mission (run each command in different cmd prompt/shells):

`./launchClient.sh -port 9000 -env`

`./launchClient.sh -port 9001 -env`

In the two agent case, running each agent in it's own shell, the run script (for agents other than the first) is given two ports 
- the first for the mission coordinator and a second (port2) for the other agent's Minecraft:

`python3 run.py --mission missions/mobchase_two_agents.xml --port 9000 --role 0 --experimentUniqueId "test1"`

`python3 run.py --mission missions/mobchase_two_agents.xml --port 9000 --port2 9001 --role 1  --experimentUniqueId "test1"`

## Running multi-threaded multi-agent examples: ##

`python3 runmultiagent.py --mission missions/mobchase_two_agents.xml 

## Installing with pip ##

If you install with `pip3 install malmoenv` then you can download the Minecraft mod 
(assuming you have git available from the command line) with: 

`python3 -c "import malmoenv.bootstrap();malmoenv.bootstrap.download()`

The sample missions will be in ./MalmoPlatform/MalmoEnv/missions.

`malmoenv.bootstrap.launchMinecraft(9000)` can be used to start up the Malmo Minecraft Mod 
listening for MalmoEnv connections on port 9000 after downloading Malmo.

## Guía rápida para este repo (Windows) ##

Requisitos:
- Python 3.7 (el mismo de los comandos bootstrap), Java 8 JDK, Git.
- Instala dependencias de `MalmoEnv/requirements.txt` y Stable-Baselines3:
  ```
  cd MalmoEnv
  py -3.7 -m pip install -r requirements.txt
  py -3.7 -m pip install stable-baselines3[extra]
  ```

Lanzar instancias de Minecraft (dos terminales):
- Terminal 1: `py -3.7 -c "import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9000)"`
- Terminal 2: `py -3.7 -c "import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9001)"`
Espera a que ambas muestren SERVER STARTED.

Entrenar multi-agente (perseguidor PPO vs escapista DQN):
- Desde `MalmoEnv`:  
  ```
  python train_chase_escape_opt.py
  ```
- Reanudar desde checkpoints específicos (opcional):  
  ```
  python train_chase_escape_opt.py ^
    --ppo-model models/Perseguidor_PPO_FINAL.zip ^
    --dqn-model models/escapista_dqn_best.zip
  ```
- Para forzar cargar *_FINAL.zip cuando también existe *_best.zip, agrega `--prefer-final`.

Evaluar modelos entrenados (solo métricas, sin seguir entrenando):
- Desde `MalmoEnv`:  
  ```
  python evaluar_modelos.py --ppo-model models/Perseguidor_PPO_FINAL.zip --dqn-model models/escapista_dqn_best.zip
  ```
  (Si omites las banderas, toma automáticamente *_best.zip y si no hay, *_FINAL.zip.)

Rutas típicas de modelos generados:
- Perseguidor PPO: `models/perseguidor_ppo_best.zip` y `models/Perseguidor_PPO_FINAL.zip`
- Escapista DQN: `models/escapista_dqn_best.zip` y `models/Escapista_DQN_FINAL.zip`

