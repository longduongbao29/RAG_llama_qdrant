FROM python:3.9-slim

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

ARG PORT=7000 
ENV PORT=$PORT

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
