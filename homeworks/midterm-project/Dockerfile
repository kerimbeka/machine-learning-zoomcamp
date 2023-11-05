FROM python:3.10.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile","Pipfile.lock","./"]

RUN pipenv install --deploy --system

COPY ["./model/model.json","./model/"]

COPY ["web_service.py","./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:9696","web_service:app"]