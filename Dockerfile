FROM python:3.9.5-slim

WORKDIR /comment-emotionality-classification

COPY req.txt req.txt

RUN apt-get update \
    && apt-get -y install libpq-dev gcc

RUN pip install -r req.txt

COPY . .

CMD ["python","-m","flask","run","--host=0.0.0.0"]