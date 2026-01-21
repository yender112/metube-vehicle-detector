# Vehicle Detection Technical Documentation

## Overview

This document describes the automatic vehicle and license plate detection system integrated into MeTube.

## Architecture

The detection pipeline consists of three main modules:

### 1. Vehicle Extraction (`vehicle_extractor.py`)

**Purpose**: Extract unique vehicles from video using YOLOv8/v11 object detection and tracking.

**Key Features**:
- Uses Ultralytics YOLO models for real-time object detection
- Tracks vehicles across frames using built-in YOLO tracker
- Detects 4 vehicle classes from COCO dataset: car, motorcycle, bus, truck
- Visual deduplication using HSV color histogram correlation
- Configurable best-frame selection strategies

**Processing Flow**:
1. Load YOLO model (auto-downloads on first run)
2. Process video with tracking enabled
3. For each tracked vehicle, store best crop based on strategy:
   - `complete`: Prioritize vehicles not touching frame edges
   - `largest`: Select frame with largest vehicle area
   - `first`: Take first valid detection (fastest)
4. Deduplicate visually similar vehicles using histogram comparison
5. Return dict of unique vehicles: `{track_id: {"image": np.array, "class": str, ...}}`

**Configuration**:
- `YOLO_MODEL`: Model size (n=nano, s=small, m=medium, l=large, x=extra-large)
- `YOLO_CONF_THRESHOLD`: Minimum detection confidence (0.0-1.0)
- `YOLO_MIN_AREA`: Minimum bounding box area in pixels
- `YOLO_SIMILARITY_THRESHOLD`: Histogram correlation threshold for deduplication
- `YOLO_STRATEGY`: Best frame selection strategy

**Performance Considerations**:
- Nano model (`yolo11n.pt`): Fastest, suitable for CPU processing
- Extra-large model (`yolo11x.pt`): Most accurate, requires GPU
- Automatically uses GPU (CUDA) if available

---

### 2. Plate Detection & OCR (`plate_filter.py`)

**Purpose**: Filter vehicles by presence of valid, readable license plates.

**Key Features**:
- Uses fast-alpr library (YOLO v9 + Vision Transformer OCR)
- Detects license plate regions in vehicle images
- Extracts text using OCR
- Validates plate format based on vehicle type
- Returns only vehicles with valid plates

**Plate Validation Rules**:
- Motorcycles: `ABC12D` pattern (3 letters, 2 numbers, 1 letter)
- Cars/Buses/Trucks: `ABC123` pattern (3 letters, 3 numbers)
- Exact 6-character length requirement

**Processing Flow**:
1. Initialize ALPR with detector and OCR models
2. For each vehicle image:
   - Detect plate regions using YOLO v9
   - Extract text using Vision Transformer OCR
   - Validate format based on vehicle class
3. Return filtered dict with only valid-plate vehicles

**Configuration**:
- `PLATE_DETECTOR_MODEL`: YOLO model for plate detection
- `PLATE_OCR_MODEL`: Vision Transformer model for text extraction

**Note**: Plate validation is strict - only perfectly formatted plates pass the filter.

---

### 3. Video Processing Queue (`video_processor.py`)

**Purpose**: Orchestrate the complete detection pipeline asynchronously.

**Key Features**:
- Sequential processing queue (one video at a time)
- Async/await pattern for non-blocking operations
- Automatic video resolution scaling for large videos
- Thread pool execution for CPU-intensive tasks
- Structured output directory creation

**Processing Pipeline**:
```
Video Download Complete
    ↓
Add to Processing Queue
    ↓
Check Video Resolution (ffprobe)
    ↓
Scale to 1920x1080 if > FHD (ffmpeg)
    ↓
Extract Vehicles (YOLO tracking) [thread pool]
    ↓
Filter by License Plate (ALPR + OCR) [thread pool]
    ↓
Save Images to {DOWNLOAD_DIR}/shots/{video_name}/
```

**Video Scaling**:
- Videos larger than 1920x1080 are automatically scaled down
- Uses H.264 codec with CRF 23 quality
- Maintains aspect ratio
- Audio track copied without re-encoding
- Original video preserved

**Output Structure**:
```
{DOWNLOAD_DIR}/
├── video.mp4              # Original download
├── video_FHD.mp4          # Scaled version (if needed)
└── shots/
    └── video/
        ├── video_car_id1_conf0.95.jpg
        ├── video_bus_id3_conf0.87.jpg
        └── video_motorcycle_id5_conf0.92.jpg
```

**Filename Format**: `{video_name}_{vehicle_class}_id{track_id}_conf{confidence}.jpg`

---

## Integration with MeTube

### Trigger Point

Detection is triggered in [ytdl.py:421-443](app/ytdl.py#L421-L443) after successful video download:

```python
if (hasattr(self.config, 'ENABLE_VEHICLE_DETECTION') and
    self.config.ENABLE_VEHICLE_DETECTION and
    download.info.status == 'finished' and
    download.info.format not in AUDIO_FORMATS):

    asyncio.create_task(
        self.video_processor.add_video(video_path, metadata)
    )
```

**Conditions**:
- `ENABLE_VEHICLE_DETECTION=true` in config
- Download completed successfully
- Not an audio-only format
- Not a chapter file (split chapters are skipped)

**Non-blocking**: Processing runs asynchronously, doesn't block new downloads.

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_VEHICLE_DETECTION` | `true` | Enable/disable detection |
| `SHOTS_DIR` | `./shots` | Output directory for detected vehicles |
| `YOLO_MODEL` | `yolo11n.pt` | YOLO model (n/s/m/l/x) |
| `YOLO_CONF_THRESHOLD` | `0.5` | Minimum detection confidence |
| `YOLO_SIMILARITY_THRESHOLD` | `0.80` | Deduplication threshold |
| `YOLO_MIN_AREA` | `40000` | Minimum vehicle area (pixels) |
| `YOLO_STRATEGY` | `complete` | Best frame strategy |
| `PLATE_DETECTOR_MODEL` | `yolo-v9-s-608-license-plate-end2end` | Plate detector |
| `PLATE_OCR_MODEL` | `global-plates-mobile-vit-v2-model` | OCR model |

### Model Storage

Models are auto-downloaded on first use:
- YOLO models: `~/.cache/ultralytics/`
- ALPR models: `~/.cache/fast-alpr/`

In Docker, these are stored in `/.cache/` (mounted as volume if persistent cache needed).

---

## Dependencies

### Python Packages (pyproject.toml)
- `ultralytics>=8.0.0`: YOLOv8/v11 implementation
- `opencv-python-headless>=4.8.0`: Computer vision operations
- `fast-alpr>=1.0.0`: License plate detection and OCR
- `numpy>=1.24.0`: Array operations

### System Libraries (Dockerfile)
- `ffmpeg`: Video processing and scaling
- `ffprobe`: Video metadata extraction
- `libgomp`, `libstdc++`, `libgcc`: OpenMP and C++ runtime
- `libjpeg-turbo`, `libpng`, `libwebp`, `tiff`: Image codecs
- `openblas`, `lapack`: Linear algebra for ML operations

---

## Performance Tuning

### For CPU-Only Systems
```yaml
environment:
  - YOLO_MODEL=yolo11n.pt        # Fastest model
  - YOLO_CONF_THRESHOLD=0.6       # Higher threshold = fewer detections
  - YOLO_MIN_AREA=50000          # Ignore small vehicles
  - YOLO_STRATEGY=first          # Skip tracking optimization
```

### For GPU Systems
```yaml
environment:
  - YOLO_MODEL=yolo11m.pt        # Balanced accuracy/speed
  - YOLO_CONF_THRESHOLD=0.4       # Lower threshold = more detections
  - YOLO_STRATEGY=complete       # Best quality crops
```

### For High Accuracy
```yaml
environment:
  - YOLO_MODEL=yolo11x.pt        # Most accurate
  - YOLO_CONF_THRESHOLD=0.3       # Catch more vehicles
  - YOLO_SIMILARITY_THRESHOLD=0.90 # Stricter deduplication
  - YOLO_MIN_AREA=30000          # Include smaller vehicles
```

---

## Troubleshooting

### No vehicles detected
- Lower `YOLO_CONF_THRESHOLD` (try 0.3)
- Reduce `YOLO_MIN_AREA` (try 20000)
- Check video quality/resolution
- Use larger model (`yolo11m.pt` or higher)

### Too many duplicate vehicles
- Increase `YOLO_SIMILARITY_THRESHOLD` (try 0.90)
- Use `complete` strategy to filter partial vehicles

### Plates not detected
- Ensure vehicle crops show clear license plate
- Check plate format matches validation rules
- Plates must be front-facing and readable
- Minimum plate resolution: ~100x30 pixels

### Performance issues
- Use smaller YOLO model (`yolo11n.pt`)
- Increase `YOLO_MIN_AREA` to skip small detections
- Use `first` strategy instead of `complete`
- Videos are automatically scaled to FHD to improve speed

---

## Logging

Detection events are logged with prefix tags:

- `[QUEUE]`: Video added to processing queue
- `[PROCESSING]`: Pipeline started for video
- `[SCALE]`: Video resolution check/scaling
- `[YOLO]`: Vehicle detection results
- `[PLATE]`: License plate filtering results
- `[SAVED]`: Final output statistics

Set `LOGLEVEL=DEBUG` for detailed processing information.

---

## Future Improvements

Possible enhancements:
- Support for custom plate formats (configurable regex)
- GPU batch processing for multiple videos
- Real-time progress notifications via WebSocket
- Plate text extraction to separate file
- Vehicle color/make/model classification
- Support for rear license plates
- Database storage for detected vehicles
