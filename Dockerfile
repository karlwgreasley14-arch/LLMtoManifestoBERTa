FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH="/app/src:/app"

# Daily collection: 02:00 UTC and 13:00 UTC
RUN echo "0 2  * * * root . /etc/environment; cd /app && python src/collector.py > /proc/1/fd/1 2>&1" >  /etc/crontab
RUN echo "0 13 * * * root . /etc/environment; cd /app && python src/collector.py > /proc/1/fd/1 2>&1" >> /etc/crontab

RUN echo '#!/bin/bash\nprintenv > /etc/environment\ncron\nexec tail -f /dev/null' > /app/run.sh
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]
