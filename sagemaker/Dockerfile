# syntax=docker/dockerfile:1
ARG REGION=us-west-2
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-inference:2.3.0-cpu-py310-ubuntu20.04-sagemaker

USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program
COPY whisperlivekit /opt/program/whisperlivekit
COPY pyproject.toml /opt/program/pyproject.toml
COPY README.md /opt/program/README.md
COPY sagemaker/inference.py /opt/program/sagemaker/inference.py

RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/program \
    SAGEMAKER_PROGRAM=sagemaker/inference.py

USER 1000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "sagemaker_inference"]
