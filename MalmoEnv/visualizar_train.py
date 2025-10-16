"""
Script para visualizar resultados del entrenamiento
Genera gráficos comparativos entre PPO y DQN
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.ndimage import uniform_filter1d

def load_monitor_data(filepath):
    """Carga datos del archivo monitor.csv"""
    try:
        df = pd.read_csv(filepath, skiprows=1)
        return df
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        return None

def smooth_curve(values, weight=0.9):
    """Suaviza una curva usando promedio exponencial"""
    if len(values) == 0:
        return []
    smoothed = []
    last = values[0]
    for val in values:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_training_comparison():
    """Genera gráficos comparativos del entrenamiento"""
    
    # Cargar datos
    ppo_data = load_monitor_data('logs/Perseguidor_PPO.monitor.csv')
    dqn_data = load_monitor_data('logs/Escapista_DQN.monitor.csv')
    
    if ppo_data is None and dqn_data is None:
        print("❌ No se encontraron datos de entrenamiento")
        print("Asegúrate de haber ejecutado train_sb3_multiagent.py primero")
        return
    
    # Crear figura con múltiples subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparación de Entrenamiento: PPO vs DQN', fontsize=16, fontweight='bold')
    
    # ===== GRÁFICO 1: Recompensas por episodio =====
    ax1 = axes[0, 0]
    
    if ppo_data is not None and 'r' in ppo_data.columns:
        episodes_ppo = range(len(ppo_data['r']))
        rewards_ppo = ppo_data['r'].values
        rewards_ppo_smooth = smooth_curve(rewards_ppo, weight=0.95)
        
        ax1.plot(episodes_ppo, rewards_ppo, alpha=0.2, color='blue', linewidth=0.5)
        ax1.plot(episodes_ppo, rewards_ppo_smooth, label='PPO (Perseguidor)', 
                color='blue', linewidth=2)
    
    if dqn_data is not None and 'r' in dqn_data.columns:
        episodes_dqn = range(len(dqn_data['r']))
        rewards_dqn = dqn_data['r'].values
        rewards_dqn_smooth = smooth_curve(rewards_dqn, weight=0.95)
        
        ax1.plot(episodes_dqn, rewards_dqn, alpha=0.2, color='red', linewidth=0.5)
        ax1.plot(episodes_dqn, rewards_dqn_smooth, label='DQN (Escapista)', 
                color='red', linewidth=2)
    
    ax1.set_xlabel('Episodio', fontsize=12)
    ax1.set_ylabel('Recompensa', fontsize=12)
    ax1.set_title('Recompensas por Episodio', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # ===== GRÁFICO 2: Longitud de episodios =====
    ax2 = axes[0, 1]
    
    if ppo_data is not None and 'l' in ppo_data.columns:
        lengths_ppo = ppo_data['l'].values
        lengths_ppo_smooth = smooth_curve(lengths_ppo, weight=0.9)
        ax2.plot(episodes_ppo, lengths_ppo_smooth, label='PPO (Perseguidor)', 
                color='blue', linewidth=2)
    
    if dqn_data is not None and 'l' in dqn_data.columns:
        lengths_dqn = dqn_data['l'].values
        lengths_dqn_smooth = smooth_curve(lengths_dqn, weight=0.9)
        ax2.plot(episodes_dqn, lengths_dqn_smooth, label='DQN (Escapista)', 
                color='red', linewidth=2)
    
    ax2.set_xlabel('Episodio', fontsize=12)
    ax2.set_ylabel('Pasos por Episodio', fontsize=12)
    ax2.set_title('Longitud de Episodios', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # ===== GRÁFICO 3: Promedio móvil de recompensas =====
    ax3 = axes[1, 0]
    window = 50  # Ventana de promedio móvil
    
    if ppo_data is not None and 'r' in ppo_data.columns:
        if len(rewards_ppo) >= window:
            moving_avg_ppo = uniform_filter1d(rewards_ppo, size=window, mode='nearest')
            ax3.plot(episodes_ppo, moving_avg_ppo, label=f'PPO (Media {window} eps)', 
                    color='blue', linewidth=2)
    
    if dqn_data is not None and 'r' in dqn_data.columns:
        if len(rewards_dqn) >= window:
            moving_avg_dqn = uniform_filter1d(rewards_dqn, size=window, mode='nearest')
            ax3.plot(episodes_dqn, moving_avg_dqn, label=f'DQN (Media {window} eps)', 
                    color='red', linewidth=2)
    
    ax3.set_xlabel('Episodio', fontsize=12)
    ax3.set_ylabel('Recompensa Promedio', fontsize=12)
    ax3.set_title(f'Promedio Móvil ({window} episodios)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # ===== GRÁFICO 4: Estadísticas finales =====
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = "ESTADÍSTICAS FINALES\n" + "="*40 + "\n\n"
    
    if ppo_data is not None and 'r' in ppo_data.columns:
        stats_text += "PPO (Perseguidor):\n"
        stats_text += f"  • Episodios: {len(ppo_data)}\n"
        stats_text += f"  • Recompensa media: {ppo_data['r'].mean():.2f}\n"
        stats_text += f"  • Recompensa máxima: {ppo_data['r'].max():.2f}\n"
        stats_text += f"  • Recompensa mínima: {ppo_data['r'].min():.2f}\n"
        stats_text += f"  • Desv. estándar: {ppo_data['r'].std():.2f}\n"
        if 'l' in ppo_data.columns:
            stats_text += f"  • Pasos promedio: {ppo_data['l'].mean():.1f}\n"
        stats_text += "\n"
    
    if dqn_data is not None and 'r' in dqn_data.columns:
        stats_text += "DQN (Escapista):\n"
        stats_text += f"  • Episodios: {len(dqn_data)}\n"
        stats_text += f"  • Recompensa media: {dqn_data['r'].mean():.2f}\n"
        stats_text += f"  • Recompensa máxima: {dqn_data['r'].max():.2f}\n"
        stats_text += f"  • Recompensa mínima: {dqn_data['r'].min():.2f}\n"
        stats_text += f"  • Desv. estándar: {dqn_data['r'].std():.2f}\n"
        if 'l' in dqn_data.columns:
            stats_text += f"  • Pasos promedio: {dqn_data['l'].mean():.1f}\n"
    
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Ajustar layout y guardar
    plt.tight_layout()
    
    # Guardar figura
    output_file = 'training_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {output_file}")
    
    # Mostrar
    plt.show()

def plot_evaluation_results():
    """Visualiza resultados de evaluación si existen"""
    
    # Buscar archivos de evaluación
    eval_files = list(Path('.').glob('evaluation_results_*.json'))
    
    if not eval_files:
        print("\n⚠ No se encontraron archivos de evaluación")
        print("Ejecuta evaluate_models.py primero")
        return
    
    # Cargar el más reciente
    latest_file = max(eval_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n📊 Visualizando: {latest_file}")
    
    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Resultados de Evaluación', fontsize=16, fontweight='bold')
    
    # Preparar datos
    agents = []
    mean_rewards = []
    std_rewards = []
    win_rates = []
    colors = ['blue', 'red']
    
    for role_str in ['0', '1']:
        if role_str in results:
            data = results[role_str]
            agents.append(data['role_name'])
            mean_rewards.append(data['mean_reward'])
            std_rewards.append(data['std_reward'])
            win_rates.append(data['win_rate'] * 100)
    
    # Gráfico 1: Recompensas promedio
    ax1 = axes[0]
    bars1 = ax1.bar(agents, mean_rewards, yerr=std_rewards, 
                    capsize=10, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Recompensa Promedio', fontsize=12)
    ax1.set_title('Recompensas en Evaluación', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar, val in zip(bars1, mean_rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')
    
    # Gráfico 2: Tasa de victoria
    ax2 = axes[1]
    bars2 = ax2.bar(agents, win_rates, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Tasa de Victoria (%)', fontsize=12)
    ax2.set_title('Tasa de Victoria', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, 
               label='50% (empate)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Añadir valores en las barras
    for bar, val in zip(bars2, win_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar
    output_file = 'evaluation_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {output_file}")
    
    plt.show()

def main():
    print("=" * 70)
    print("  VISUALIZACIÓN DE RESULTADOS")
    print("=" * 70)
    
    # Verificar que existan datos
    logs_exist = Path('logs').exists()
    eval_exists = len(list(Path('.').glob('evaluation_results_*.json'))) > 0
    
    if not logs_exist and not eval_exists:
        print("\n❌ No se encontraron datos para visualizar")
        print("\nAsegúrate de haber ejecutado:")
        print("  1. train_sb3_multiagent.py (para generar logs/)")
        print("  2. evaluate_models.py (para generar evaluation_results_*.json)")
        return
    
    print("\nGenerando gráficos...\n")
    
    # Gráficos de entrenamiento
    if logs_exist:
        print("📈 Visualizando datos de entrenamiento...")
        try:
            plot_training_comparison()
        except Exception as e:
            print(f"Error generando gráficos de entrenamiento: {e}")
    
    # Gráficos de evaluación
    if eval_exists:
        print("\n📊 Visualizando resultados de evaluación...")
        try:
            plot_evaluation_results()
        except Exception as e:
            print(f"Error generando gráficos de evaluación: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Visualización completada!")
    print("=" * 70)

if __name__ == '__main__':
    # Verificar dependencias
    try:
        import matplotlib
        import pandas
        import scipy
    except ImportError as e:
        print("❌ Faltan dependencias para visualización:")
        print(f"   {e}")
        print("\nInstala con:")
        print("   pip install matplotlib pandas scipy")
        exit(1)
    
    main()