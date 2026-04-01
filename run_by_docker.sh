#!/usr/bin/env sh
set -e

IMAGE_NAME="diemdanh:latest"
CONTAINER_NAME="diemdanh"
PORT_MAPPING="8000:8000"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not on PATH." >&2
  exit 1
fi

echo "Building image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

if docker ps -a --format '{{.Names}}' | grep -x "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Removing existing container: $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME"
fi

echo "Starting container: $CONTAINER_NAME"
docker run -d \
  --name "$CONTAINER_NAME" \
  -p "$PORT_MAPPING" \
  "$IMAGE_NAME"

echo "Done. API should be available at http://localhost:8000"
