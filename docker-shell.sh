#!/bin/bash

set -e

# Create the network if we don't have it yet
docker network inspect cinns_network_base >/dev/null 2>&1 || docker network create cinns_network_base

# Build the image based on the Dockerfile
docker build -t cinns_network_base -f Dockerfile .

# Run All Containers
docker-compose run --rm --service-ports cinns_network_base