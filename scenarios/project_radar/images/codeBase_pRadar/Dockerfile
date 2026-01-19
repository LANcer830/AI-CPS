FROM python:3.9-slim
WORKDIR /app
COPY code /app
RUN pip install pandas tensorflow
CMD ["python", "apply_model.py"]
