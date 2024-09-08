FROM nvidia/cuda:12.1.0-base-ubuntu20.04

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

COPY . .
RUN pip install uvicorn fastapi
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 9875

# Start the app
CMD ["uvicorn", "main2:app", "--host", "0.0.0.0", "--port", "9875"]