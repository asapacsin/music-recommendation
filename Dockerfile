# 1️⃣ Base image: slim Python for cost + security
FROM python:3.11-slim

# 2️⃣ Environment settings (production-safe)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3️⃣ System dependencies (only what is needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Set working directory
WORKDIR /app

# 5️⃣ Copy dependency list FIRST (Docker cache optimization)
COPY requirements.txt .

# 6️⃣ Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7️⃣ Copy application code LAST
COPY app ./app

# 8️⃣ Copy local dependencies
COPY dependency ./dependency

# 9️⃣ Create folders
RUN mkdir -p ./data/input \
    && mkdir -p ./data/movie_input \
    && mkdir -p ./data/music_db

# 🔧 Install local libraries

RUN apt-get update && apt-get install -y git

WORKDIR /app/dependency

RUN echo "starting installation"

RUN for d in */ ; do \
      pip install -e "$d"; \
    done

# 10️⃣ Expose API port
EXPOSE 8000

# 11️⃣ Run FastAPI with Uvicorn
WORKDIR /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]