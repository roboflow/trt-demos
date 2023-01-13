# Roboflow TRT Container Demos

This repository is for quickly accessing examples on how to use the Roboflow TRT Docker container. Our docker container is an enterprise feature, please refer to our [Enterprise Documentation](https://docs.roboflow.com/inference/enterprise/) for more information.

## Install Roboflow

To install this package, please use `Python 3.6` or higher. We provide three different ways to install the Roboflow
package to use within your own projects.

Install from PyPi (Recommended):

```bash
pip install roboflow
```

## Building the Roboflow Container Load Balencer

To build the load balancer docker container use the command below.

```
docker build . -t lb
```

Make sure that the names of the services in the docker-compose.yaml file are correctly reflected in the .conf/roboflow-nginx.conf file.

```
docker-compose up
```

Your Docker should now be spinning up multiple GPU containers that all share a volume and a port with the load balancer. This way the load balencer can manage the throughput of each container for optimal speed.

# Configuring Testing Scripts

Before we start docker and run the scripts we need to configure them to your Roboflow account first. For this next step we will need your Roboflow API Key. If you don't have your API key, you can learn how to get it in our [REST API documentation.](https://docs.roboflow.com/rest-api#obtaining-your-api-key)

## Configure TRT-API-MultiContainer.py Test Script

```python
# Setting up arrays for Roboflow model varaibles (Lines 15-18)
version_number_array = [version_num1, version_num2, version_num3, version_num4]
project_id_array = ["model_id1/", "model_id2/", "model_id3/", "model_id4/"]
images_folder_array = ["folder_path1", "folder_path2", "folder_path3", "folder_path4"]
api_key_array = ['API1', 'API2', 'API3', 'API4'] # It is likely you will have the same API keys
```


## Quickstart - GPU Guide

Make sure that you have Docker Desktop running or if you are using Linux you can start Docker Daemon using `sudo service docker start` while having Docker installed.

These examples will require two terminal windows. Open them now for your convenience.


### Run inference with API (Recommended)

```bash
## Terminal 1 - Run docker ('sudo' linux only)
sudo docker run --gpus all -p 9001:9001 roboflow/inference-server-trt:latest
## Terminal 2
python3 TRT-API.py
```

### Run inference with Python Package

```bash
## Terminal 1 - Run docker ('sudo' linux only)
sudo docker run --gpus all -p 9001:9001 roboflow/inference-server-trt:latest
## Terminal 2
python3 TRT-Python.py
```

### Run inference on multiple containers with API

```bash
## Terminal 1 - Run docker ('sudo' linux only)
sudo docker compose up
## Terminal 2
python3 TRT-API-MultiContainer.py
```
