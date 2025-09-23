import malmoenv
import argparse
from pathlib import Path
import numpy as np
import cv2
import time
import os
from collections import deque
import matplotlib.pyplot as plt


class VisionDebugger:
    """🔥 Analizador visual completo para entender qué ve el agente"""

    def __init__(self, save_images=True, show_live=True):
        self.save_images = save_images
        self.show_live = show_live
        self.frame_count = 0
        self.color_history = deque(maxlen=100)
        self.brightness_history = deque(maxlen=100)

        # Crear directorio para guardar análisis
        if save_images:
            os.makedirs("debug_analysis", exist_ok=True)
            os.makedirs("debug_analysis/frames", exist_ok=True)
            os.makedirs("debug_analysis/regions", exist_ok=True)

    def analyze_frame(self, obs, step, agent_pos=None, action_taken=None):
        """Análisis completo de un frame"""
        try:
            if obs is None or obs.size == 0:
                print(f"❌ Frame {step}: Observación vacía")
                return None

            h, w, d = 360, 640, 3

            if obs.size != h * w * d:
                print(f"❌ Frame {step}: Tamaño incorrecto {obs.size}, esperado {h * w * d}")
                return None

            img = obs.reshape(h, w, d)
            self.frame_count += 1

            # 🔥 ANÁLISIS DETALLADO
            analysis = self._comprehensive_analysis(img, step, agent_pos, action_taken)

            # Guardar imagen anotada
            if self.save_images and step % 5 == 0:  # Cada 5 frames
                self._save_annotated_frame(img, analysis, step)

            # Mostrar en vivo
            if self.show_live:
                self._show_live_analysis(img, analysis, step)

            return analysis

        except Exception as e:
            print(f"❌ Error analizando frame {step}: {e}")
            return None

    def _comprehensive_analysis(self, img, step, agent_pos, action_taken):
        """Análisis comprehensivo de la imagen"""
        h, w = img.shape[:2]
        center_h, center_w = h // 2, w // 2

        analysis = {
            'step': step,
            'agent_pos': agent_pos,
            'action_taken': action_taken,
            'overall_stats': {},
            'regions': {},
            'colors': {},
            'recommendations': []
        }

        # 1. 🔥 ESTADÍSTICAS GENERALES
        analysis['overall_stats'] = {
            'brightness_mean': float(np.mean(img)),
            'brightness_std': float(np.std(img)),
            'brightness_min': float(np.min(img)),
            'brightness_max': float(np.max(img)),
            'dominant_color': self._get_dominant_color(img),
            'color_variance': float(np.var(img))
        }

        # 2. 🔥 ANÁLISIS DE REGIONES CRÍTICAS
        regions = {
            'front_immediate': img[center_h - 8:center_h + 8, center_w - 12:center_w + 12],
            'front_far': img[center_h - 15:center_h + 15, center_w - 20:center_w + 20],
            'left_side': img[center_h - 10:center_h + 10, center_w - 40:center_w - 20],
            'right_side': img[center_h - 10:center_h + 10, center_w + 20:center_w + 40],
            'top_view': img[0:60, center_w - 30:center_w + 30],
            'bottom_view': img[h - 60:h, center_w - 30:center_w + 30],
            'left_corner': img[center_h - 20:center_h + 20, 0:40],
            'right_corner': img[center_h - 20:center_h + 20, w - 40:w],
        }

        for region_name, region in regions.items():
            if region.size > 0:
                analysis['regions'][region_name] = {
                    'brightness': float(np.mean(region)),
                    'std': float(np.std(region)),
                    'size': region.shape,
                    'is_clear': self._is_path_clear(region),
                    'dominant_color': self._get_dominant_color(region),
                    'texture_complexity': float(np.std(cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)))
                }

        # 3. 🔥 DETECCIÓN DE ELEMENTOS ESPECÍFICOS
        analysis['colors'] = self._detect_special_blocks(img)

        # 4. 🔥 RECOMENDACIONES AUTOMÁTICAS
        analysis['recommendations'] = self._generate_recommendations(analysis)

        # Guardar en historial
        self.brightness_history.append(analysis['overall_stats']['brightness_mean'])

        return analysis

    def _is_path_clear(self, region):
        """Determina si una región representa un camino libre"""
        brightness = np.mean(region)
        std_dev = np.std(region)

        # Lógica mejorada para detectar caminos
        if brightness > 100:  # Muy brillante = probablemente glowstone (camino)
            return True
        elif brightness < 50:  # Muy oscuro = probablemente piedra (pared)
            return False
        elif std_dev < 20:  # Muy uniforme = probablemente pared sólida
            return False
        else:
            return brightness > 70  # Umbral moderado

    def _get_dominant_color(self, region):
        """Obtiene el color dominante en una región"""
        try:
            # Convertir a formato para análisis
            pixels = region.reshape(-1, 3)

            # Calcular color promedio
            avg_color = np.mean(pixels, axis=0)

            # Clasificar el color
            r, g, b = avg_color

            if r > 200 and g > 200 and b < 100:  # Amarillo (glowstone)
                return "yellow_glowstone"
            elif r < 80 and g < 80 and b < 80:  # Oscuro (piedra)
                return "dark_stone"
            elif r > 150 and g < 100 and b < 100:  # Rojo (redstone)
                return "red_block"
            elif r > 200 and g > 150 and b < 100:  # Dorado (gold)
                return "gold_block"
            elif r > 200 and g > 200 and b > 200:  # Blanco (cuarzo)
                return "white_quartz"
            else:
                return f"rgb({int(r)},{int(g)},{int(b)})"

        except:
            return "unknown"

    def _detect_special_blocks(self, img):
        """Detecta bloques especiales en la imagen"""
        colors = {}

        # Definir rangos de colores para bloques especiales
        color_ranges = {
            'redstone_block': ([100, 0, 0], [255, 50, 50]),  # Rojo
            'gold_block': ([150, 120, 0], [255, 255, 100]),  # Dorado
            'glowstone': ([200, 200, 100], [255, 255, 255]),  # Amarillo brillante
            'stone': ([30, 30, 30], [80, 80, 80]),  # Gris oscuro
            'quartz': ([200, 200, 200], [255, 255, 255]),  # Blanco
        }

        for block_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(img, np.array(lower), np.array(upper))
            pixel_count = cv2.countNonZero(mask)
            percentage = (pixel_count / (img.shape[0] * img.shape[1])) * 100

            colors[block_name] = {
                'pixel_count': int(pixel_count),
                'percentage': float(percentage),
                'detected': percentage > 1.0  # Más del 1% de la imagen
            }

        return colors

    def _generate_recommendations(self, analysis):
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []

        regions = analysis['regions']
        colors = analysis['colors']

        # 🔥 RECOMENDACIONES DE NAVEGACIÓN
        if 'front_immediate' in regions:
            front = regions['front_immediate']
            if front['is_clear']:
                recommendations.append("✅ ADELANTE: Camino despejado, seguro avanzar")
            else:
                recommendations.append("🚧 ADELANTE: Obstáculo detectado, considerar girar")

        if 'left_side' in regions and 'right_side' in regions:
            left_clear = regions['left_side']['is_clear']
            right_clear = regions['right_side']['is_clear']

            if left_clear and right_clear:
                recommendations.append("🔀 OPCIONES: Tanto izquierda como derecha están libres")
            elif left_clear:
                recommendations.append("⬅️ RECOMENDACIÓN: Girar a la izquierda")
            elif right_clear:
                recommendations.append("➡️ RECOMENDACIÓN: Girar a la derecha")
            else:
                recommendations.append("🚨 ALERTA: Posible callejón sin salida")

        # 🔥 DETECCIÓN DE OBJETIVOS
        if colors.get('redstone_block', {}).get('detected', False):
            recommendations.append("🎯 ¡OBJETIVO FINAL VISIBLE! Redstone block detectado")

        if colors.get('gold_block', {}).get('detected', False):
            recommendations.append("💰 ¡CHECKPOINT VISIBLE! Gold block detectado")

        # 🔥 ANÁLISIS DE CONTEXTO
        brightness = analysis['overall_stats']['brightness_mean']
        if brightness > 150:
            recommendations.append("🔆 ÁREA BRILLANTE: Probablemente espacio abierto o glowstone")
        elif brightness < 60:
            recommendations.append("🌑 ÁREA OSCURA: Probablemente cerca de paredes de piedra")

        return recommendations

    def _save_annotated_frame(self, img, analysis, step):
        """Guarda el frame con anotaciones visuales"""
        try:
            # Crear copia para anotar
            annotated = img.copy()
            h, w = img.shape[:2]
            center_h, center_w = h // 2, w // 2

            # 🔥 DIBUJAR REGIONES DE ANÁLISIS
            # Frente inmediato (rojo)
            cv2.rectangle(annotated, (center_w - 12, center_h - 8), (center_w + 12, center_h + 8), (255, 0, 0), 2)
            cv2.putText(annotated, "FRONT", (center_w - 12, center_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0),
                        1)

            # Frente lejano (rojo claro)
            cv2.rectangle(annotated, (center_w - 20, center_h - 15), (center_w + 20, center_h + 15), (255, 100, 100), 1)
            cv2.putText(annotated, "FAR", (center_w - 20, center_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 100, 100), 1)

            # Izquierda (verde)
            cv2.rectangle(annotated, (center_w - 40, center_h - 10), (center_w - 20, center_h + 10), (0, 255, 0), 2)
            cv2.putText(annotated, "LEFT", (center_w - 38, center_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),
                        1)

            # Derecha (azul)
            cv2.rectangle(annotated, (center_w + 20, center_h - 10), (center_w + 40, center_h + 10), (0, 0, 255), 2)
            cv2.putText(annotated, "RIGHT", (center_w + 22, center_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                        1)

            # 🔥 INFORMACIÓN EN PANTALLA
            info_lines = [
                f"Step: {step}",
                f"Brightness: {analysis['overall_stats']['brightness_mean']:.1f}",
                f"Agent: {analysis.get('agent_pos', 'Unknown')}",
                f"Action: {analysis.get('action_taken', 'None')}"
            ]

            for i, line in enumerate(info_lines):
                cv2.putText(annotated, line, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Estado de regiones
            y_pos = 150
            for region_name, region_data in analysis['regions'].items():
                if region_name in ['front_immediate', 'left_side', 'right_side']:
                    status = "CLEAR" if region_data['is_clear'] else "BLOCKED"
                    color = (0, 255, 0) if region_data['is_clear'] else (0, 0, 255)
                    text = f"{region_name}: {status}"
                    cv2.putText(annotated, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_pos += 15

            # Guardar imagen
            filename = f"debug_analysis/frames/frame_{step:04d}.png"
            cv2.imwrite(filename, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            # También guardar análisis en texto
            self._save_analysis_text(analysis, step)

        except Exception as e:
            print(f"❌ Error guardando frame anotado: {e}")

    def _save_analysis_text(self, analysis, step):
        """Guarda el análisis detallado en texto"""
        try:
            filename = f"debug_analysis/analysis_{step:04d}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"=== ANÁLISIS FRAME {step} ===\n\n")

                f.write("📍 INFORMACIÓN GENERAL:\n")
                f.write(f"   Posición agente: {analysis.get('agent_pos', 'Desconocida')}\n")
                f.write(f"   Acción tomada: {analysis.get('action_taken', 'Ninguna')}\n\n")

                f.write("📊 ESTADÍSTICAS DE IMAGEN:\n")
                stats = analysis['overall_stats']
                for key, value in stats.items():
                    f.write(f"   {key}: {value}\n")
                f.write("\n")

                f.write("🔍 ANÁLISIS DE REGIONES:\n")
                for region_name, region_data in analysis['regions'].items():
                    f.write(f"   {region_name}:\n")
                    for key, value in region_data.items():
                        f.write(f"      {key}: {value}\n")
                    f.write("\n")

                f.write("🎨 DETECCIÓN DE COLORES:\n")
                for color_name, color_data in analysis['colors'].items():
                    if color_data['detected']:
                        f.write(f"   ✅ {color_name}: {color_data['percentage']:.1f}% de la imagen\n")

                f.write("\n💡 RECOMENDACIONES:\n")
                for rec in analysis['recommendations']:
                    f.write(f"   {rec}\n")

        except Exception as e:
            print(f"❌ Error guardando análisis en texto: {e}")

    def _show_live_analysis(self, img, analysis, step):
        """Muestra análisis en vivo"""
        try:
            # Crear imagen para mostrar
            display_img = img.copy()
            h, w = img.shape[:2]
            center_h, center_w = h // 2, w // 2

            # Dibujar regiones básicas
            cv2.rectangle(display_img, (center_w - 12, center_h - 8), (center_w + 12, center_h + 8), (255, 0, 0), 1)
            cv2.rectangle(display_img, (center_w - 40, center_h - 10), (center_w - 20, center_h + 10), (0, 255, 0), 1)
            cv2.rectangle(display_img, (center_w + 20, center_h - 10), (center_w + 40, center_h + 10), (0, 0, 255), 1)

            # Info básica
            cv2.putText(display_img, f"Step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, f"Brightness: {analysis['overall_stats']['brightness_mean']:.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Mostrar
            cv2.imshow('Agent Vision Debug', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        except Exception as e:
            print(f"❌ Error en visualización en vivo: {e}")

    def print_summary(self, analysis):
        """Imprime resumen del análisis en consola"""
        if not analysis:
            return

        print(f"\n🔍 === ANÁLISIS VISUAL STEP {analysis['step']} ===")

        # Estadísticas generales
        stats = analysis['overall_stats']
        print(f"📊 Brillo: {stats['brightness_mean']:.1f} (±{stats['brightness_std']:.1f})")
        print(f"🎨 Color dominante: {stats['dominant_color']}")

        # Estado de regiones críticas
        regions = analysis['regions']
        if 'front_immediate' in regions:
            front_status = "🟢 LIBRE" if regions['front_immediate']['is_clear'] else "🔴 BLOQUEADO"
            print(f"⬆️ Frente: {front_status} (brillo: {regions['front_immediate']['brightness']:.1f})")

        if 'left_side' in regions and 'right_side' in regions:
            left_status = "🟢" if regions['left_side']['is_clear'] else "🔴"
            right_status = "🟢" if regions['right_side']['is_clear'] else "🔴"
            print(f"⬅️ Izquierda: {left_status} | Derecha: {right_status} ➡️")

        # Detecciones especiales
        colors = analysis['colors']
        special_detected = []
        for color_name, color_data in colors.items():
            if color_data['detected']:
                special_detected.append(f"{color_name} ({color_data['percentage']:.1f}%)")

        if special_detected:
            print(f"🎯 Bloques especiales: {', '.join(special_detected)}")

        # Recomendaciones principales
        if analysis['recommendations']:
            print("💡 Recomendación principal:", analysis['recommendations'][0])

    def generate_session_report(self):
        """Genera un reporte de toda la sesión de debug"""
        try:
            report_path = "debug_analysis/session_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== REPORTE DE SESIÓN DEBUG ===\n\n")
                f.write(f"Total de frames analizados: {self.frame_count}\n")

                if self.brightness_history:
                    f.write(f"Brillo promedio de sesión: {np.mean(self.brightness_history):.1f}\n")
                    f.write(f"Variación de brillo: {np.std(self.brightness_history):.1f}\n")

                f.write("\n📁 Archivos generados:\n")
                f.write(f"   - Frames anotados: debug_analysis/frames/\n")
                f.write(f"   - Análisis detallados: debug_analysis/analysis_*.txt\n")
                f.write(f"   - Este reporte: {report_path}\n")

            print(f"📋 Reporte de sesión guardado en: {report_path}")

        except Exception as e:
            print(f"❌ Error generando reporte: {e}")


# 🔥 FUNCIÓN PRINCIPAL PARA TESTING DE VISIÓN
def debug_agent_vision(xml_file, episodes=5, steps_per_episode=50):
    """Función principal para debuggear la visión del agente"""

    print("🎯 === MODO DEBUG DE VISIÓN DEL AGENTE ===")
    print("Este modo te permitirá ver exactamente qué percibe el agente\n")

    # Inicializar debugger
    vision_debugger = VisionDebugger(save_images=True, show_live=True)

    try:
        xml = Path(xml_file).read_text()
        env = malmoenv.make()
        env.init(xml, 9000, server='127.0.0.1')

        for episode in range(episodes):
            print(f"\n🎮 === EPISODIO DEBUG {episode + 1}/{episodes} ===")
            obs = env.reset()

            for step in range(steps_per_episode):
                try:
                    # Acción simple para movimiento
                    if step % 10 == 0:
                        action = 0  # Avanzar
                    elif step % 10 < 3:
                        action = 1  # Girar derecha
                    elif step % 10 < 6:
                        action = 2  # Girar izquierda
                    else:
                        action = 0  # Avanzar

                    # Ejecutar paso
                    next_obs, reward, done, info = env.step(action)

                    # Obtener posición del agente
                    agent_pos = None
                    if info and isinstance(info, list) and len(info) > 0:
                        pos_info = info[0]
                        if isinstance(pos_info, dict) and 'XPos' in pos_info and 'ZPos' in pos_info:
                            agent_pos = (pos_info['XPos'], pos_info['ZPos'])

                    # 🔥 ANÁLISIS VISUAL COMPLETO
                    analysis = vision_debugger.analyze_frame(
                        obs, step, agent_pos, f"Action_{action}"
                    )

                    # Mostrar resumen cada 5 steps
                    if step % 5 == 0 and analysis:
                        vision_debugger.print_summary(analysis)

                    obs = next_obs
                    time.sleep(0.1)  # Pausa para poder ver

                    if done:
                        print(f"   ✅ Episodio completado en step {step}")
                        break

                except Exception as e:
                    print(f"❌ Error en step {step}: {e}")
                    break

        # Generar reporte final
        vision_debugger.generate_session_report()

    except Exception as e:
        print(f"❌ Error crítico en debug: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
            cv2.destroyAllWindows()
        except:
            pass
        print("🔌 Debug completado")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug Vision Analyzer para Maze Runner')
    parser.add_argument('--mission', type=str, default='missions/mazerunner5x5.xml')
    parser.add_argument('--episodes', type=int, default=3, help='Episodios para debug')
    parser.add_argument('--steps', type=int, default=30, help='Steps por episodio')

    args = parser.parse_args()

    print("🔍 Iniciando análisis de visión del agente...")
    print("📁 Los resultados se guardarán en: debug_analysis/")
    print("👁️ Ventana en vivo: 'Agent Vision Debug'")
    print("⏹️ Presiona Ctrl+C para detener\n")

    debug_agent_vision(args.mission, args.episodes, args.steps)