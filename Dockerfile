FROM python:3.14

RUN groupadd --gid 1000 appuser \
 && useradd  --uid 1000 --gid 1000 --create-home appuser \
 && mkdir -p /home/appuser/app \
 && chown -R appuser:appuser /home/appuser/app

USER appuser

WORKDIR /home/appuser/app
