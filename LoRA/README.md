# BERT Demo with LoRA

This repository contains a demo of BERT with Low-Rank Adaptation (LoRA). This README will guide you through building the Docker image and running it.

## Prerequisites

- Docker installed on your machine

## Building the Docker Image

To build the Docker image, navigate to the directory containing the `Dockerfile` and run the following command:

```sh
docker build -t bert-lora-demo .
```

## Running the Docker Image

Once the image is built, you can run it using the following command:

```sh
docker run -it --rm bert-lora-demo
```

This command will start a container from the `bert-lora-demo` image and open an interactive terminal.

## Additional Information

For more details on how to use the demo, please refer to the documentation within the repository.
