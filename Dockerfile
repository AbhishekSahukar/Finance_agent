# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Build React frontend
# ══════════════════════════════════════════════════════════════════════════════
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# Install dependencies first (cached layer if package files unchanged)
COPY frontend/package*.json ./
RUN npm ci

# Copy source and build
COPY frontend/ ./
RUN npm run build
# Output: /app/frontend/dist/


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Python runtime + FastAPI
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS final

# System deps needed by PyMuPDF (libmupdf) and standard build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# ── Backend source ────────────────────────────────────────────────────────────
COPY backend/ ./backend/

# ── Built frontend (from Stage 1) ────────────────────────────────────────────
COPY --from=frontend-build /app/frontend/dist/ ./frontend/dist/

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# Render / Railway / Fly inject $PORT at runtime; default 8000 for local Docker
ENV PORT=8000

EXPOSE 8000

# Run from backend/ so relative imports and logging.yaml path resolution work
CMD ["sh", "-c", "cd /app/backend && uvicorn main:app --host 0.0.0.0 --port $PORT"]