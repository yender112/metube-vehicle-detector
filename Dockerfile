FROM node:lts-alpine AS builder

WORKDIR /metube
COPY ui ./
RUN corepack enable && corepack prepare pnpm --activate
RUN pnpm install && pnpm run build


FROM python:3.13-alpine

WORKDIR /app

COPY pyproject.toml uv.lock docker-entrypoint.sh ./

# Use sed to strip carriage-return characters from the entrypoint script (in case building on Windows)
# Install dependencies
RUN sed -i 's/\r$//g' docker-entrypoint.sh && \
    chmod +x docker-entrypoint.sh && \
    # Install system dependencies for OpenCV and ML libraries
    apk add --update ffmpeg aria2 coreutils shadow su-exec curl tini deno gdbm-tools sqlite file \
        # OpenCV dependencies
        libgomp libstdc++ libgcc \
        # Image processing libraries
        libjpeg-turbo libpng libwebp tiff openblas lapack && \
    # Install build dependencies
    apk add --update --virtual .build-deps gcc g++ musl-dev uv \
        # OpenCV build dependencies
        jpeg-dev libpng-dev libwebp-dev tiff-dev openblas-dev lapack-dev && \
    # Install Python packages with uv
    UV_PROJECT_ENVIRONMENT=/usr/local uv sync --frozen --no-dev --compile-bytecode && \
    # Clean up build dependencies
    apk del .build-deps && \
    rm -rf /var/cache/apk/* && \
    mkdir /.cache && chmod 777 /.cache

COPY app ./app
COPY --from=builder /metube/dist/metube ./ui/dist/metube

ENV UID=1000
ENV GID=1000
ENV UMASK=022

ENV DOWNLOAD_DIR /downloads
ENV STATE_DIR /downloads/.metube
ENV TEMP_DIR /downloads
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

ENTRYPOINT ["/sbin/tini", "-g", "--", "./docker-entrypoint.sh"]
