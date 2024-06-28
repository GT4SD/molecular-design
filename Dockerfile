# Base image containing the installed gt4sd environment
#FROM drugilsberg/gt4sd-base:v1.4.2-cpu
FROM quay.io/gt4sd/gt4sd-base:v1.4.2-cpu


# Certs for git clone
RUN apt-get update && \
    apt-get install -y git ca-certificates && \
    apt-get clean
# Clone this repository
RUN git clone --branch add-docker https://github.com/GT4SD/molecular-design.git

WORKDIR /workspace/molecular-design

RUN echo $API_KEY
RUN ls

# We need to use the pypi toxsmi package, not the default one
RUN pip uninstall --yes toxsmi && pip install toxsmi && mkdir data

RUN chmod +x example_pipeline.sh




CMD ["bash"]
