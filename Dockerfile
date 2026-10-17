FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 USE_GPU=false
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE requirements.txt ./
COPY src ./src
RUN python -m pip install --no-cache-dir .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://localhost:8000/health'))['is_ready']"
CMD ["python","-m","uvicorn","src.api:app","--host","0.0.0.0","--port","8000","--workers","1"]
