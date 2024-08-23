FROM python:3.9-slim

RUN apt-get update && apt-get install -y gcc build-essential

WORKDIR /app
COPY . /app
COPY start.sh /app/start.sh
RUN pip install -r requirements.txt

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]

