FROM python:3.10.4-alpine3.15

WORKDIR /comment-emotionality-classification

COPY req.txt req.txt

RUN pip install -r req.txt

COPY . .

CMD ["python","-m","flask","run","--host=0.0.0.0"]