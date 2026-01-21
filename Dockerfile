FROM node:lts-alpine AS builder

WORKDIR /metube
COPY ui ./
RUN corepack enable && corepack prepare pnpm --activate
RUN pnpm install && pnpm run build


FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml uv.lock docker-entrypoint.sh ./

# Use sed to strip carriage-return characters from the entrypoint script (in case building on Windows)
# Install dependencies
RUN sed -i 's/\r$//g' docker-entrypoint.sh && \
    chmod +x docker-entrypoint.sh && \
    # Install system dependencies
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        aria2 \
        curl \
        tini \
        gosu \
        file \
        # OpenCV and image processing dependencies
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        # Image format libraries
        libjpeg62-turbo \
        libpng16-16 \
        libwebp-dev \
        libtiff-dev \
        libopenblas0 \
        liblapack3 && \
    # Install uv for Python package management
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    # Add uv to PATH and install Python packages
    export PATH="/root/.local/bin:$PATH" && \
    uv lock --upgrade && \
    UV_PROJECT_ENVIRONMENT=/usr/local uv sync --no-dev --compile-bytecode && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache && \
    mkdir /.cache && chmod 777 /.cache

COPY app ./app
COPY --from=builder /metube/dist/metube ./ui/dist/metube

ENV UID=1000
ENV GID=1000
ENV UMASK=022

ENV DOWNLOAD_DIR=/downloads
ENV STATE_DIR=/downloads/.metube
ENV TEMP_DIR=/downloads
VOLUME /downloads
EXPOSE 8081

# Vehicle and plate detection configuration
ENV ENABLE_VEHICLE_DETECTION=true
ENV SHOTS_DIR=/downloads/shots
ENV YOLO_MODEL=yolo11n.pt
ENV YOLO_CONF_THRESHOLD=0.5
ENV YOLO_SIMILARITY_THRESHOLD=0.80
ENV YOLO_MIN_AREA=40000
ENV YOLO_STRATEGY=complete
ENV PLATE_DETECTOR_MODEL=yolo-v9-s-608-license-plate-end2end
ENV PLATE_OCR_MODEL=global-plates-mobile-vit-v2-model

# Add build-time argument for version
ARG VERSION=dev
ENV METUBE_VERSION=$VERSION

ENTRYPOINT ["/usr/bin/tini", "-g", "--", "./docker-entrypoint.sh"]
