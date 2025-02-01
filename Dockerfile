# FROM ubuntu:20.04
# FROM python:3.9

# WORKDIR /usr/src/app

# COPY requirements.txt ./

# RUN pip install --no-cache-dir -r requirements.txt

# COPY ./system ./system

# # COPY ./dataset/mnist ./dataset/mnist

# # COPY ./results ./results

# # RUN chmod +x ./system/run_sh/mnist1.sh

# # ENTRYPOINT ["sh", "./system/run_sh/mnist1.sh"]

# CMD ["python3", "./system/main.py"]

# # ENTRYPOINT ["sh", "./mnist1.sh"]


# COPY src dest: src host local machine which we install docker image and dest container 
# /dest/ this is root directory
# dest/ this is not the root directory but WORKDIR/dest/

# LABEL key="value" information provider such as author, email

# EXPOSE port for access container


#############################################
# FROM sydat2701/almednet:v1.8

# RUN apt update; \
# apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev; \
# wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz; \
# tar -zxvf Python-3.9.7.tgz; \
# cd Python-3.9.7; \
# ./configure --prefix=/usr/local/python3; \
# make && make install; \
# ln -sf /usr/local/python3/bin/python3.9 /usr/bin/python3; \
# ln -sf /usr/local/python3/bin/pip3.9 /usr/bin/pip3

# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

############################################

FROM trongbinh246/docker-test:v5

# WORKDIR /app

# COPY requirements.txt .

RUN pip3 install opacus
RUN pip3 install wandb
RUN pip3 install calmsize
RUN pip install cvxpy