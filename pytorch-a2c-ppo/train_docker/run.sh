#!/bin/bash

exec docker run -it --mount src="$(pwd)",target=/code,type=bind train:latest "$@"