FROM python:3.8.2

RUN mkdir /workspace
WORKDIR /workspace

COPY . /workspace

RUN pip install -r requirements.txt
