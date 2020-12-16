FROM python:3.7.6
ADD ./ /
WORKDIR /
RUN apt update
RUN apt-get install -y cmake
RUN apt-get install -y libgl1-mesa-glx
RUN pip install -r requirements.txt
CMD ["gunicorn", "app:app"]
