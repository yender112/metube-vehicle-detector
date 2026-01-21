"""
Sistema de procesamiento de videos con cola secuencial.
Extrae vehículos únicos y filtra por presencia de placa válida.
"""

import asyncio
import logging
import subprocess
import shutil
import time
import cv2
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    from . import vehicle_extractor
    from . import plate_filter as plate_filter_module
    from . import file_mover
except ImportError:
    import vehicle_extractor
    import plate_filter as plate_filter_module
    import file_mover

log = logging.getLogger('video_processor')


@dataclass
class ProcessingInfo:
    """Tracks the processing status of a video."""
    id: str
    video_path: str
    title: str
    url: str = ''
    filename: str = ''
    status: str = 'pending'  # pending, scaling, extracting, filtering, saving, moving, completed, error
    percent: int = 0
    current_step: str = ''
    error: Optional[str] = None
    timestamp: int = field(default_factory=lambda: time.time_ns())
    vehicles_detected: int = 0
    vehicles_with_plates: int = 0
    shots_saved: int = 0
    download_dir: str = ''

    def to_dict(self):
        return asdict(self)


class ProcessingQueueNotifier:
    """Abstract notifier for processing status updates."""

    async def processing_added(self, info: ProcessingInfo):
        raise NotImplementedError

    async def processing_updated(self, info: ProcessingInfo):
        raise NotImplementedError

    async def processing_completed(self, info: ProcessingInfo):
        raise NotImplementedError

    async def processing_error(self, info: ProcessingInfo):
        raise NotImplementedError


class VideoProcessingQueue:
    """Cola de procesamiento secuencial para videos."""

    def __init__(self, config, notifier: Optional[ProcessingQueueNotifier] = None):
        """
        Args:
            config: Objeto de configuración con atributos para YOLO y ALPR
            notifier: Optional notifier for status updates
        """
        self.queue = asyncio.Queue()
        self.processing_lock = asyncio.Lock()
        self.worker_task = None
        self.config = config
        self.notifier = notifier

        # Track processing status
        self.processing = {}  # id -> ProcessingInfo
        self.completed = {}   # id -> ProcessingInfo

        # File mover for SMB transfers
        self.file_mover = file_mover.FileMover(config)

    async def add_video(self, video_path, metadata):
        """
        Encola video para procesamiento.

        Args:
            video_path: Ruta absoluta al video descargado
            metadata: Dict con información del video (title, url, format, etc.)
        """
        # Create processing info
        info = ProcessingInfo(
            id=metadata.get('url', video_path),
            video_path=video_path,
            title=metadata.get('title', Path(video_path).stem),
            url=metadata.get('url', ''),
            filename=metadata.get('filename', ''),
            download_dir=metadata.get('download_dir', '.')
        )

        self.processing[info.id] = info

        log.info(f"[QUEUE] Video encolado: {metadata['title']}")
        await self.queue.put({"path": video_path, "metadata": metadata, "info": info})

        if self.notifier:
            await self.notifier.processing_added(info)

        # Iniciar worker si no está corriendo
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())

    async def _update_status(self, info: ProcessingInfo, status: str, percent: int, step: str = ''):
        """Update processing status and notify."""
        info.status = status
        info.percent = percent
        info.current_step = step
        if self.notifier:
            await self.notifier.processing_updated(info)

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
        Pipeline completo de procesamiento: escalado → extracción → filtrado → guardado → mover.

        Args:
            item: Dict con 'path' (video), 'metadata' e 'info'
        """
        video_path = item["path"]
        metadata = item["metadata"]
        info = item["info"]
        video_name = Path(video_path).stem

        log.info(f"[PROCESSING] Iniciando: {metadata['title']}")

        try:
            # Directorio de salida: {download_dir}/shots/{nombreDelVideo}/
            download_dir = metadata.get('download_dir', '.')
            shots_dir = Path(download_dir) / 'shots' / video_name
            shots_dir.mkdir(parents=True, exist_ok=True)

            loop = asyncio.get_event_loop()

            # Paso 1: Verificar resolución y escalar si es necesario (0-10%)
            await self._update_status(info, 'scaling', 5, 'Checking video resolution')
            processing_video = await loop.run_in_executor(
                None,
                self._scale_video_if_needed,
                video_path
            )

            if processing_video != video_path:
                log.info(f"[SCALE] Video escalado a FHD: {processing_video}")
            await self._update_status(info, 'scaling', 10, 'Video ready for processing')

            # Paso 2: Extraer vehículos (10-50%)
            await self._update_status(info, 'extracting', 15, 'Loading YOLO model')
            crops_dict = await loop.run_in_executor(
                None,
                vehicle_extractor.extract_vehicles_to_dict,
                processing_video,
                self.config
            )

            info.vehicles_detected = len(crops_dict)
            log.info(f"[YOLO] {len(crops_dict)} vehículos detectados")
            await self._update_status(info, 'extracting', 50, f'{len(crops_dict)} vehicles detected')

            if len(crops_dict) == 0:
                log.info(f"[PROCESSING] No se detectaron vehículos en {metadata['title']}")
                await self._update_status(info, 'completed', 100, 'No vehicles detected')
                self._move_to_completed(info)
                if self.notifier:
                    await self.notifier.processing_completed(info)
                return

            # Paso 3: Filtrar por placa (50-80%)
            await self._update_status(info, 'filtering', 55, 'Initializing plate recognition')
            plate_filter_instance = await loop.run_in_executor(
                None,
                plate_filter_module.PlateFilter,
                self.config.PLATE_DETECTOR_MODEL,
                self.config.PLATE_OCR_MODEL
            )

            await self._update_status(info, 'filtering', 60, 'Filtering vehicles by plate')
            filtered_crops = await loop.run_in_executor(
                None,
                plate_filter_module.filter_crops_by_plate,
                crops_dict,
                plate_filter_instance
            )

            info.vehicles_with_plates = len(filtered_crops)
            log.info(f"[PLATE] {len(filtered_crops)} vehículos con placa válida")
            await self._update_status(info, 'filtering', 80, f'{len(filtered_crops)} vehicles with valid plates')

            # Paso 4: Guardar imágenes (80-90%)
            await self._update_status(info, 'saving', 85, 'Saving vehicle images')
            saved = 0
            for track_id, data in filtered_crops.items():
                filename = f"{video_name}_{data['class']}_id{track_id}_conf{data['conf']:.2f}.jpg"
                filepath = shots_dir / filename
                cv2.imwrite(str(filepath), data['image'])
                saved += 1

            info.shots_saved = saved
            log.info(f"[SAVED] {saved} imágenes guardadas en {shots_dir}")
            await self._update_status(info, 'saving', 90, f'{saved} images saved')

            # Paso 5: Mover a SMB si está habilitado (90-100%)
            if self.file_mover.is_enabled():
                await self._update_status(info, 'moving', 95, 'Moving files to network share')
                move_result = await loop.run_in_executor(
                    None,
                    self.file_mover.move_to_smb,
                    video_path,
                    str(shots_dir),
                    metadata['title']
                )
                if move_result['status'] == 'success':
                    log.info(f"[SMB] Files moved to: {move_result['destination']}")
                else:
                    log.warning(f"[SMB] Move failed: {move_result.get('msg', 'Unknown error')}")

            # Completado
            await self._update_status(info, 'completed', 100, f'{saved} images saved')
            self._move_to_completed(info)
            if self.notifier:
                await self.notifier.processing_completed(info)

        except Exception as e:
            log.error(f"[ERROR] Fallo procesando {video_path}: {e}", exc_info=True)
            info.error = str(e)
            await self._update_status(info, 'error', info.percent, str(e))
            self._move_to_completed(info)
            if self.notifier:
                await self.notifier.processing_error(info)

    def _move_to_completed(self, info: ProcessingInfo):
        """Move processing info from processing to completed dict."""
        if info.id in self.processing:
            del self.processing[info.id]
        self.completed[info.id] = info

    async def retry_processing(self, id: str) -> dict:
        """
        Retry a failed processing job.

        Args:
            id: The processing ID to retry

        Returns:
            dict with status
        """
        if id not in self.completed:
            return {'status': 'error', 'msg': 'Processing job not found'}

        info = self.completed[id]
        if info.status != 'error':
            return {'status': 'error', 'msg': 'Processing job is not in error state'}

        # Remove from completed
        del self.completed[id]

        # Re-add with original metadata
        metadata = {
            'title': info.title,
            'url': info.url,
            'filename': info.filename,
            'download_dir': info.download_dir
        }

        await self.add_video(info.video_path, metadata)
        return {'status': 'ok'}

    def get_status(self) -> dict:
        """Get all processing statuses."""
        return {
            'processing': [info.to_dict() for info in self.processing.values()],
            'completed': [info.to_dict() for info in self.completed.values()]
        }

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
