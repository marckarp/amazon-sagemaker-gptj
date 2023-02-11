#!/usr/bin/env bash

IMAGE_NAME=$1

export IMAGE_NAME=$IMAGE_NAME
pwd
### Build container--------
cd container
chmod +x src/serve
docker build -t $IMAGE_NAME .


IMAGE_ID="$(docker inspect --format="{{.Id}}" $IMAGE_NAME)"

echo $IMAGE_ID

docker images




