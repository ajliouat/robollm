FROM python:3.11-slim

# MuJoCo rendering dependencies (OSMesa for headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglew-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["pytest", "tests/", "-v"]
