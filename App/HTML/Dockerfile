FROM python:3.7

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         nginx \
    && rm -rf /var/lib/apt/lists/*
	
COPY conf/nginx.conf /etc/nginx/sites-enabled/default

RUN pip install pandas scikit-learn flask uwsgi django

