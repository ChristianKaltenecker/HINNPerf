# HINNPerf relies on tensorflow. Typically, such applications need GPU capabilities,
# but we cannot confirm this for HINNPerf.
# Therefore, we use a container image based on Debian 11 and python3.7
FROM docker.io/python:3.7-bullseye 

WORKDIR /application

# Update the whole package
RUN apt update

RUN apt upgrade -y

RUN apt install -y git

# Install package dependencies
#RUN pip install tensorflow==1.15.3

# Degrade protobuf version
#RUN pip install protobuf==3.20.3

#RUN pip install pandas==1.3.5

#RUN pip install scikit-learn==1.0.2

RUN git clone https://github.com/ChristianKaltenecker/HINNPerf.git

RUN pip install -r HINNPerf/requirements.txt
