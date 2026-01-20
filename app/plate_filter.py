"""
Filtro de vehículos por presencia de placa visible.
Puede usarse independiente o como módulo.

Requiere: pip install fast-alpr opencv-python
"""

import cv2
import re
import argparse
import shutil
from pathlib import Path
from fast_alpr import ALPR


def is_valid_plate(text: str, vehicle_type: str = None) -> bool:
    """Valida formato de placa según tipo de vehículo."""
    if not text or len(text) != 6:
        return False
    text = text.upper()
    if vehicle_type == "motorcycle":
        return bool(re.match(r'^[A-Z]{3}[0-9]{2}[A-Z]$', text))
    elif vehicle_type in {"car", "bus", "truck"}:
        return bool(re.match(r'^[A-Z]{3}[0-9]{3}$', text))
    return len(text) == 6


class PlateFilter:
    """Filtra imágenes de vehículos según si tienen placa visible y válida."""

    def __init__(
        self,
        detector_model: str = "yolo-v9-s-608-license-plate-end2end",
        ocr_model: str = "global-plates-mobile-vit-v2-model"
    ):
        """
        Args:
            detector_model: Modelo de detección de placas
            ocr_model: Modelo OCR para leer texto
        """
        self.alpr = ALPR(
            detector_model=detector_model,
            ocr_model=ocr_model
        )
        print(f"ALPR inicializado: detector={detector_model}")

    def has_plate(self, image, vehicle_type: str = None, return_text: bool = False):
        """
        Verifica si la imagen contiene una placa visible y válida.

        Args:
            image: numpy array (BGR) o ruta a imagen
            vehicle_type: "car", "motorcycle", "bus", "truck" para validación
            return_text: Si True, retorna (bool, texto_placa)

        Returns:
            True si se detecta placa válida, False si no
            O tupla (bool, str) si return_text=True
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                return (False, None) if return_text else False

        # Convertir BGR a RGB para fast_alpr
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plate_results = self.alpr.predict(image_rgb)

        for result in plate_results:
            text = result.ocr.text if result.ocr else None

            if text and is_valid_plate(text, vehicle_type):
                return (True, text.upper()) if return_text else True

        return (False, None) if return_text else False

    def filter_directory(
        self,
        input_dir: str,
        output_dir: str = None,
        delete_no_plate: bool = False,
        copy_mode: bool = True,
        save_plate_text: bool = False
    ) -> dict:
        """
        Filtra imágenes de un directorio por presencia de placa válida.

        Args:
            input_dir: Carpeta con imágenes de vehículos
            output_dir: Carpeta para imágenes con placa (None = mismo directorio)
            delete_no_plate: Si True, elimina imágenes sin placa
            copy_mode: Si True copia, si False mueve
            save_plate_text: Si True, guarda archivo con textos de placas

        Returns:
            dict con estadísticas
        """
        input_path = Path(input_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        images = [f for f in input_path.iterdir()
                  if f.suffix.lower() in image_extensions]

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path

        stats = {"total": len(images), "with_plate": 0, "without_plate": 0}
        plate_texts = []

        for i, img_file in enumerate(images):
            image = cv2.imread(str(img_file))
            if image is None:
                continue

            # Inferir tipo de vehículo del nombre del archivo
            filename_lower = img_file.name.lower()
            vehicle_type = None
            for vtype in ["motorcycle", "car", "bus", "truck"]:
                if vtype in filename_lower:
                    vehicle_type = vtype
                    break

            has_plate, text = self.has_plate(image, vehicle_type=vehicle_type, return_text=True)

            if has_plate:
                stats["with_plate"] += 1
                plate_texts.append(f"{img_file.name}: {text}")

                if output_dir:
                    dest = output_path / img_file.name
                    if copy_mode:
                        shutil.copy2(img_file, dest)
                    else:
                        shutil.move(img_file, dest)
            else:
                stats["without_plate"] += 1
                if delete_no_plate:
                    img_file.unlink()

            if (i + 1) % 50 == 0:
                print(f"Procesadas {i+1}/{len(images)} imágenes")

        # Guardar textos de placas
        if save_plate_text and plate_texts:
            txt_path = output_path / "placas_detectadas.txt"
            with open(txt_path, "w") as f:
                f.write("\n".join(plate_texts))
            print(f"Textos de placas guardados en: {txt_path}")

        print(f"\nResultados: {stats['with_plate']} con placa válida, {stats['without_plate']} sin placa válida")
        return stats


def filter_crops_by_plate(crops_dict: dict, plate_filter: PlateFilter) -> dict:
    """
    Filtra diccionario de crops eliminando los que no tienen placa válida.
    Para usar desde vehicle_extractor.py

    Args:
        crops_dict: {track_id: {"image": np.array, "class": str, ...}}
        plate_filter: Instancia de PlateFilter

    Returns:
        Diccionario filtrado solo con vehículos que tienen placa válida
    """
    filtered = {}
    for track_id, data in crops_dict.items():
        if data["image"] is not None:
            vehicle_type = data.get("class")
            if plate_filter.has_plate(data["image"], vehicle_type=vehicle_type):
                filtered[track_id] = data
    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filtrar vehículos por presencia de placa válida")
    parser.add_argument("input", help="Carpeta con imágenes de vehículos")
    parser.add_argument("-o", "--output", help="Carpeta de salida (default: filtrar en sitio)")
    parser.add_argument("--delete", action="store_true",
                        help="Eliminar imágenes sin placa válida")
    parser.add_argument("--move", action="store_true",
                        help="Mover en vez de copiar")
    parser.add_argument("--save-text", action="store_true",
                        help="Guardar archivo con textos de placas detectadas")

    args = parser.parse_args()

    pf = PlateFilter()
    pf.filter_directory(
        args.input,
        output_dir=args.output,
        delete_no_plate=args.delete,
        copy_mode=not args.move,
        save_plate_text=args.save_text
    )
