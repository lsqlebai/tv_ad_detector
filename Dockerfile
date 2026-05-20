FROM python:3.11-slim

WORKDIR /work

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir -r /work/requirements.txt

COPY . /work

EXPOSE 8787

CMD ["python", "scripts/web_app.py", "--host", "0.0.0.0", "--port", "8787"]
