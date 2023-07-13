# HINNPerf relies on tensorflow. Typically, such applications need GPU capabilities,
# but we cannot confirm this for HINNPerf.
# Therefore, we use a container image based on Debian 11 and python3.7
FROM python:3.7-bullseye 

WORKDIR /application

# Update the whole package
RUN apt update

RUN apt upgrade

RUN apt install git

# Install package dependencies
RUN pip install tensorflow==1.15.3

# Degrade protobuf version
RUN pip install protobuf==3.20.3

RUN git clone https://github.com/ChristianKaltenecker/HINNPerf.git

