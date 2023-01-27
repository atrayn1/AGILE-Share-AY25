# Basic Dockerfile to run the AGILE streamlit app
# agile/Dockerfile

FROM python:3.9-slim

WORKDIR /agile

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git ssh \
    && rm -rf /var/lib/apt/lists/*

# Take private key as argument
ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/
RUN echo "$SSH_PRIVATE_KEY" > /root/.ssh/id_rsa
RUN chmod -R 600 /root/.ssh/id_rsa

# make sure your domain is accepted
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# Remove the -branch when merged to main
RUN git clone --branch docker git@github.com:Sam-Chanow/AGILE.git

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
