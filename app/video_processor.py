"""
Sistema de procesamiento de videos con cola secuencial.
Extrae vehículos únicos y filtra por presencia de placa válida.
"""

import asyncio
import logging
import subprocess
import cv2
import os
from pathlib import Path
try:
    from . import vehicle_extractor
    from . import plate_filter as plate_filter_module
except ImportError:
    import vehicle_extractor
    import plate_filter as plate_filter_module

log = logging.getLogger('video_processor')


class VideoProcessingQueue:
    """Cola de procesamiento secuencial para videos."""

    def __init__(self, config):
        """
        Args:
            config: Objeto de configuración con atributos para YOLO y ALPR
        """
        self.queue = asyncio.Queue()
        self.processing_lock = asyncio.Lock()
        self.worker_task = None
        self.config = config

    async def add_video(self, video_path, metadata):
        """
        Encola video para procesamiento.

        Args:
            video_path: Ruta absoluta al video descargado
            metadata: Dict con información del video (title, url, format, etc.)
        """
        log.info(f"[QUEUE] Video encolado: {metadata['title']}")
        await self.queue.put({"path": video_path, "metadata": metadata})

        # Iniciar worker si no está corriendo
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())

    async def _worker(self):
        """Worker que procesa cola secuencialmente."""
        while True:
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self.queue.empty():
                    break  # Salir si no hay más trabajo
                continue

            # LOCK: Solo 1 video se procesa a la vez
            async with self.processing_lock:
                await self._process_single_video(item)

    async def _process_single_video(self, item):
        """
        Pipeline completo de procesamiento: escalado → extracción → filtrado → guardado.

        Args:
            item: Dict con 'path' (video) y 'metadata'
        """
        video_path = item["path"]
        metadata = item["metadata"]
        video_name = Path(video_path).stem

        log.info(f"[PROCESSING] Iniciando: {metadata['title']}")

        try:
            # Directorio de salida: {download_dir}/shots/{nombreDelVideo}/
            download_dir = metadata.get('download_dir', '.')
            shots_dir = Path(download_dir) / 'shots' / video_name
            shots_dir.mkdir(parents=True, exist_ok=True)

            # Paso 1: Verificar resolución y escalar si es necesario
            loop = asyncio.get_event_loop()
            processing_video = await loop.run_in_executor(
                None,
                self._scale_video_if_needed,
                video_path
            )

            if processing_video != video_path:
                log.info(f"[SCALE] Video escalado a FHD: {processing_video}")

            # Paso 2: Extraer vehículos (ejecutar en thread pool)
            crops_dict = await loop.run_in_executor(
                None,
                vehicle_extractor.extract_vehicles_to_dict,
                processing_video,  # Usar video escalado o original
                self.config
            )

            log.info(f"[YOLO] {len(crops_dict)} vehículos detectados")

            if len(crops_dict) == 0:
                log.info(f"[PROCESSING] No se detectaron vehículos en {metadata['title']}")
                return

            # Paso 3: Filtrar por placa (ejecutar en thread pool)
            plate_filter_instance = await loop.run_in_executor(
                None,
                plate_filter_module.PlateFilter,
                self.config.PLATE_DETECTOR_MODEL,
                self.config.PLATE_OCR_MODEL
            )

            filtered_crops = await loop.run_in_executor(
                None,
                plate_filter_module.filter_crops_by_plate,
                crops_dict,
                plate_filter_instance
            )

            log.info(f"[PLATE] {len(filtered_crops)} vehículos con placa válida")

            # Paso 4: Guardar imágenes
            saved = 0
            for track_id, data in filtered_crops.items():
                filename = f"{video_name}_{data['class']}_id{track_id}_conf{data['conf']:.2f}.jpg"
                filepath = shots_dir / filename
                cv2.imwrite(str(filepath), data['image'])
                saved += 1

            log.info(f"[SAVED] {saved} imágenes guardadas en {shots_dir}")

        except Exception as e:
            log.error(f"[ERROR] Fallo procesando {video_path}: {e}", exc_info=True)

    def _scale_video_if_needed(self, video_path: str) -> str:
        """
        Verifica resolución del video y escala a 1920x1080 si es mayor.

        Args:
            video_path: Ruta al video original

        Returns:
            str: Path al video a procesar (original o escalado)
        """
        # Obtener resolución del video con ffprobe
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            video_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            width, height = map(int, result.stdout.strip().split(','))

            log.info(f"[SCALE] Resolución detectada: {width}x{height}")

            # Si es menor o igual a 1920x1080, usar original
            if width <= 1920 and height <= 1080:
                log.info(f"[SCALE] Video ya está en FHD o menor, no se escala")
                return video_path

            # Escalar a 1920x1080
            video_dir = Path(video_path).parent
            video_stem = Path(video_path).stem
            video_ext = Path(video_path).suffix
            fhd_path = video_dir / f"{video_stem}_FHD{video_ext}"

            log.info(f"[SCALE] Escalando {width}x{height} → 1920x1080")

            # ffmpeg: escalar manteniendo aspect ratio, max 1920x1080
            scale_cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease',
                '-c:v', 'libx264',  # Codec H.264
                '-preset', 'fast',   # Balance velocidad/calidad
                '-crf', '23',        # Calidad constante
                '-c:a', 'copy',      # Copiar audio sin recodificar
                '-y',                # Sobrescribir si existe
                str(fhd_path)
            ]

            subprocess.run(scale_cmd, check=True, capture_output=True)

            log.info(f"[SCALE] Video escalado guardado: {fhd_path}")
            return str(fhd_path)

        except subprocess.CalledProcessError as e:
            log.error(f"[SCALE] Error escalando video: {e.stderr}")
            return video_path  # Usar original si falla
        except Exception as e:
            log.error(f"[SCALE] Error inesperado: {e}")
            return video_path
