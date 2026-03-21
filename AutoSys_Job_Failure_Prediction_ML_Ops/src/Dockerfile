FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/

RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

COPY ./app
EXPOSE 5001
CMD ["python", "app.py"]