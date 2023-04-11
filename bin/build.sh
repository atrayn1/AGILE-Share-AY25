#!/bin/bash

build () {
    echo "BUILDING AGILE"
    echo "Verifying Docker Installation"

    if [ -x "$(command -v docker)" ]; then
        docker build -t agile .
    else
        echo "Build failed"
        echo "Docker Installation not located, please install docker before proceeding."
        echo "Installation instructions can be found here: https://www.docker.com/products/docker-desktop/"
    fi
    
    echo "DONE"
}

run () {
    echo "Running dockerized AGILE"
    docker run -p 8501:8501 agile
    echo "DONE"
}
