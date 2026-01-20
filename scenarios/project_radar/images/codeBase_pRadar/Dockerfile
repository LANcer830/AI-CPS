FROM python:3.9-slim

WORKDIR /app

# Install stable, compatible stack
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy inference code
COPY images/codeBase_pRadar/code/ /app/

CMD ["python", "apply_model.py"]