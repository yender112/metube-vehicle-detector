"""
Extractor de vehículos únicos de videos usando YOLOv8 + tracking.
Requiere: pip install ultralytics opencv-python
"""

import cv2
import os
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict


# Clases COCO relevantes para vehículos
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


def compute_histogram(image, bins=64):
    """Calcula histograma de color normalizado para comparación."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def is_duplicate(new_crop, existing_crops, threshold=0.85):
    """
    Verifica si new_crop es similar a alguno de los crops existentes.
    Usa correlación de histogramas.
    """
    if len(existing_crops) == 0:
        return False

    new_hist = compute_histogram(new_crop)

    for existing in existing_crops:
        existing_hist = compute_histogram(existing["image"])
        # Correlación: 1.0 = idénticos, 0 = completamente diferentes
        similarity = cv2.compareHist(new_hist, existing_hist, cv2.HISTCMP_CORREL)
        if similarity > threshold:
            return True

    return False


def deduplicate_crops(crops_dict, similarity_threshold=0.85):
    """
    Elimina duplicados visuales del diccionario de crops.
    Agrupa por clase y compara dentro de cada clase.
    """
    # Agrupar por clase
    by_class = defaultdict(list)
    for track_id, data in crops_dict.items():
        if data["image"] is not None:
            by_class[data["class"]].append({"track_id": track_id, **data})

    # Deduplicar dentro de cada clase
    unique_crops = {}
    for cls_name, crops in by_class.items():
        # Ordenar por área (mayor primero) para quedarnos con los mejores
        crops.sort(key=lambda x: x["area"], reverse=True)

        kept = []
        for crop in crops:
            if not is_duplicate(crop["image"], kept, similarity_threshold):
                kept.append(crop)
                unique_crops[crop["track_id"]] = {
                    "image": crop["image"],
                    "area": crop["area"],
                    "class": crop["class"],
                    "conf": crop["conf"]
                }

    return unique_crops


def extract_vehicles_to_dict(video_path: str, config) -> dict:
    """
    Versión que retorna dict en memoria en lugar de guardar archivos.
    Usada por video_processor.py

    Args:
        video_path: Ruta al video
        config: Objeto de configuración con atributos YOLO_MODEL, YOLO_CONF_THRESHOLD, etc.

    Returns:
        {track_id: {"image": np.array, "area": int, "class": str, "conf": float}}
    """
    # Cargar modelo (automáticamente usa GPU si está disponible)
    model = YOLO(config.YOLO_MODEL)
    try:
        model.to('cuda')
    except:
        pass  # Si no hay GPU, se queda en CPU

    # Verificar si usa GPU
    device = model.device
    print(f"Usando dispositivo: {device}")

    # Almacenar el mejor crop por cada track_id
    best_crops = defaultdict(lambda: {"area": 0, "image": None, "class": None, "conf": 0, "is_complete": False})

    conf_threshold = float(config.YOLO_CONF_THRESHOLD)
    min_area = int(config.YOLO_MIN_AREA)
    best_frame_strategy = config.YOLO_STRATEGY

    # Procesar video con tracking integrado de YOLO
    results = model.track(
        source=video_path,
        classes=list(VEHICLE_CLASSES.keys()),  # Solo vehículos
        conf=conf_threshold,
        stream=True,
        persist=True,
        verbose=False,
        device=0 if device.type == 'cuda' else 'cpu'
    )

    frame_count = 0
    for result in results:
        frame_count += 1

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes

        # Verificar que hay tracking IDs
        if boxes.id is None:
            continue

        frame = result.orig_img
        h, w = frame.shape[:2]

        for i, box in enumerate(boxes):
            track_id = int(box.id[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Obtener coordenadas del bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Calcular área
            area = (x2 - x1) * (y2 - y1)

            # Ignorar detecciones muy pequeñas
            if area < min_area:
                continue

            # Verificar si el vehículo está completo (no toca los bordes)
            margin_border = 10
            is_complete = (
                x1 >= margin_border and
                y1 >= margin_border and
                x2 <= w - margin_border and
                y2 <= h - margin_border
            )

            # Decidir si guardar este crop
            current = best_crops[track_id]
            should_save = False

            if best_frame_strategy == "first":
                should_save = current["image"] is None
            elif best_frame_strategy == "largest":
                should_save = area > current["area"]
            else:  # complete
                # Prioridad: completo > incompleto, luego por área
                current_is_complete = current.get("is_complete", False)

                if is_complete and not current_is_complete:
                    should_save = True
                elif is_complete == current_is_complete:
                    should_save = area > current["area"]

            if should_save:
                # Extraer crop con margen de seguridad
                margin = 5
                x1_safe = max(0, x1 - margin)
                y1_safe = max(0, y1 - margin)
                x2_safe = min(w, x2 + margin)
                y2_safe = min(h, y2 + margin)

                crop = frame[y1_safe:y2_safe, x1_safe:x2_safe].copy()

                best_crops[track_id] = {
                    "image": crop,
                    "area": area,
                    "class": VEHICLE_CLASSES[cls_id],
                    "conf": conf,
                    "is_complete": is_complete
                }

        if frame_count % 100 == 0:
            print(f"Procesados {frame_count} frames, {len(best_crops)} vehículos detectados")

    print(f"\nDeduplicando visualmente...")
    similarity_threshold = float(config.YOLO_SIMILARITY_THRESHOLD)
    unique_crops = deduplicate_crops(best_crops, similarity_threshold=similarity_threshold)
    print(f"Reducido de {len(best_crops)} a {len(unique_crops)} vehículos únicos")

    return unique_crops


def extract_vehicles(
    video_path: str,
    output_dir: str,
    model_name: str = "yolo11m.pt",
    conf_threshold: float = 0.5,
    best_frame_strategy: str = "complete",
    similarity_threshold: float = 0.80,
    min_area: int = 40000
):
    """
    Extrae un recuadro por cada vehículo único detectado en el video.
    Versión original CLI que guarda archivos directamente.

    Args:
        video_path: Ruta al video
        output_dir: Carpeta de salida para los recuadros
        model_name: Modelo YOLO a usar (yolo11n/s/m/l/x.pt)
        conf_threshold: Umbral de confianza mínimo
        best_frame_strategy: "complete" (vehículo entero), "largest", o "first"
        similarity_threshold: Umbral para considerar dos crops como duplicados (0.0-1.0)
        min_area: Área mínima en píxeles
    """
    # Crear objeto config simulado para reusar extract_vehicles_to_dict
    class Config:
        YOLO_MODEL = model_name
        YOLO_CONF_THRESHOLD = conf_threshold
        YOLO_STRATEGY = best_frame_strategy
        YOLO_SIMILARITY_THRESHOLD = similarity_threshold
        YOLO_MIN_AREA = min_area

    config = Config()

    # Extraer vehículos
    unique_crops = extract_vehicles_to_dict(video_path, config)

    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem

    # Guardar todos los crops
    saved_count = 0
    for track_id, data in unique_crops.items():
        if data["image"] is not None:
            filename = f"{video_name}_{data['class']}_id{track_id}_conf{data['conf']:.2f}.jpg"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), data["image"])
            saved_count += 1

    print(f"\nCompletado: {saved_count} vehículos únicos guardados")
    print(f"Guardados en: {output_path}")

    return saved_count


def process_multiple_videos(video_dir: str, output_dir: str, **kwargs):
    """Procesa múltiples videos de una carpeta."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_dir = Path(video_dir)

    total = 0
    for video_file in video_dir.iterdir():
        if video_file.suffix.lower() in video_extensions:
            print(f"\n{'='*50}")
            print(f"Procesando: {video_file.name}")
            print('='*50)
            count = extract_vehicles(str(video_file), output_dir, **kwargs)
            total += count

    print(f"\n{'='*50}")
    print(f"TOTAL: {total} vehículos extraídos de todos los videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraer vehículos únicos de videos")
    parser.add_argument("input", help="Video o carpeta de videos")
    parser.add_argument("-o", "--output", default="vehiculos_extraidos", help="Carpeta de salida")
    parser.add_argument("-m", "--model", default="yolo11n.pt",
                        help="Modelo YOLO (yolo11n/s/m/l/x.pt)")
    parser.add_argument("-c", "--conf", type=float, default=0.5, help="Umbral de confianza")
    parser.add_argument("-s", "--similarity", type=float, default=0.80,
                        help="Umbral de similitud para deduplicación (0.0-1.0, mayor=más estricto)")
    parser.add_argument("--min-area", type=int, default=40000,
                        help="Área mínima en píxeles (default: 40000 = 200x200)")
    parser.add_argument("--strategy", choices=["largest", "first", "complete"], default="complete",
                        help="Estrategia: 'complete' (vehículo entero), 'largest' (mayor área), 'first' (más rápido)")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        process_multiple_videos(
            args.input, args.output,
            model_name=args.model,
            conf_threshold=args.conf,
            best_frame_strategy=args.strategy,
            similarity_threshold=args.similarity,
            min_area=args.min_area
        )
    else:
        extract_vehicles(
            args.input, args.output,
            model_name=args.model,
            conf_threshold=args.conf,
            best_frame_strategy=args.strategy,
            similarity_threshold=args.similarity,
            min_area=args.min_area
        )
