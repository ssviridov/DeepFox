FROM griver/aai_base:latest

WORKDIR /repos
RUN git clone https://github.com/openai/baselines.git
RUN cd baselines && pip install -e .

COPY ./ /repos/pytorch-a2c-ppo
RUN cd /repos/pytorch-a2c-ppo && pip install -e .

WORKDIR /code

ENV HTTP_PROXY ""
ENV HTTPS_PROXY ""
ENV http_proxy ""
ENV https_proxy ""

ENTRYPOINT ["python", "train_aai.py"]

