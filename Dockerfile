FROM python:3.6-slim

ARG WORK_DIR="/home/polyaxon"

WORKDIR $WORK_DIR

RUN mkdir /home/polyaxon/src
COPY requirements.txt /home/polyaxon
COPY src/. /home/polyaxon/src
COPY tensorfood_small.h5 /home/polyaxon

WORKDIR $WORK_DIR

RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "src/app.py"]