FROM python:3.12-slim

LABEL authors="Duarte Folgado <duarte.folgado@fraunhofer.pt>"

WORKDIR /app/

COPY requirements/requirements.txt requirements/requirements-prod.txt /app/requirements/
RUN pip install --no-cache-dir -r /app/requirements/requirements.txt && pip install --no-cache-dir -r /app/requirements/requirements-prod.txt

# copy code and models
# COPY models /app/models
COPY src /app/src
COPY .env /app/.env

ENV PYTHONPATH="${PYTHONPATH}:/app/src/mscmivida"

# The code to run when container is started
CMD ["python", "src/mscmivida/api.py"]
