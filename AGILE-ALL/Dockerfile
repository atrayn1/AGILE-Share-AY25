# Basic Dockerfile to run the AGILE streamlit app
# agile/Dockerfile

FROM python:3.8-slim

WORKDIR /agile

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git ssh \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install torch==2.4.0 pandas==1.3.5 numpy==1.21.6 

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]