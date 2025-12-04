# debug_bootstrap.py
import malmoenv.bootstrap
import os
import sys

print("="*60)
print("DEBUG: malmoenv.bootstrap")
print("="*60)

# 1. Ubicación
bootstrap_file = malmoenv.bootstrap.__file__
print(f"\n1. Archivo: {bootstrap_file}")
bootstrap_dir = os.path.dirname(bootstrap_file)
print(f"   Directorio: {bootstrap_dir}")

# 2. Contenido del directorio
print(f"\n2. Archivos en malmoenv:")
for item in os.listdir(bootstrap_dir):
    full_path = os.path.join(bootstrap_dir, item)
    if os.path.isfile(full_path):
        size = os.path.getsize(full_path)
        print(f"   - {item} ({size} bytes)")
    else:
        print(f"   - {item}/ (carpeta)")

# 3. Buscar archivos clave
print(f"\n3. Archivos críticos:")
critical_files = ['launchClient.bat', 'launchClient.sh', 'gradlew.bat', 
                  'launch_minecraft_in_background.py', 'build.gradle']
for fname in critical_files:
    fpath = os.path.join(bootstrap_dir, fname)
    exists = "OK" if os.path.exists(fpath) else "X"
    print(f"   {exists} {fname}")

# 4. Ver código de launch_minecraft
print(f"\n4. Código de launch_minecraft:")
import inspect
try:
    source = inspect.getsource(malmoenv.bootstrap.launch_minecraft)
    print(source[:500])  # Primeras 500 caracteres
except:
    print("   No se pudo obtener el código")

print("\n" + "="*60)
