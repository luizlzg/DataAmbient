FROM python:3.10-buster

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container
COPY . /app

RUN apt-get update -y && apt-get -y install nano python3-dev

# Install any dependencies
RUN pip install --upgrade pip setuptools wheel
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


CMD ["bash"]