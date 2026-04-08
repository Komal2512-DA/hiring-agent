FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

COPY requirements.txt /code/requirements.txt
RUN pip install --upgrade pip && pip install -r /code/requirements.txt

COPY . /code

EXPOSE 7860

ENV ENABLE_WEB_INTERFACE=true
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

