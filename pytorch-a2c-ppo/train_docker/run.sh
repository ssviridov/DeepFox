#!/bin/bash

exec docker run --runtime=nvidia --rm -it --mount src="$(pwd)",target=/code,type=bind train:latest "$@"