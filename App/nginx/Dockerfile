FROM python:3.7

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         nginx \
    && rm -rf /var/lib/apt/lists/*
	
COPY conf/nginx.conf /etc/nginx/sites-enabled/default

RUN pip install pandas scikit-learn flask uwsgi
RUN pip3 install django==3.0.4
RUN pip3 install gunicorn==20.0.2
RUN pip3 install Werkzeug