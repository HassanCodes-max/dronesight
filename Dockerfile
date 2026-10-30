FROM python:3.12-slim

RUN useradd --create-home --uid 1000 app
WORKDIR /srv
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
COPY dronesight dronesight
COPY models models
COPY demo demo
COPY start.sh .
RUN mkdir -p data/app && chown -R app:app data

USER app
ENV DRONESIGHT_ENV=production DRONESIGHT_DATA=/srv/data/app
EXPOSE 7860
CMD ["sh", "start.sh"]
