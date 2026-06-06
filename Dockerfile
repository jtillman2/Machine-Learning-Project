FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Python deps first so this layer is cached between code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Chromium + all OS-level dependencies Playwright needs
RUN playwright install chromium --with-deps

# Copy the rest of the app
COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
