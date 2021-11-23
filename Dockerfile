# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt && apt-get update && apt -y install rsync grsync

COPY . .

ENV PATH="/app:${PATH}"