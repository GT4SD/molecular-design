# Base image containing the installed gt4sd environment
#FROM drugilsberg/gt4sd-base:v1.4.2-cpu
FROM quay.io/gt4sd/gt4sd-base:v1.5.0-cpu

# Since there is no v1.5.1 base-image tag, we upgrade GT4SD in place.
RUN pip install --no-deps --no-cache-dir gt4sd==1.5.1

# Certs for git clone
RUN apt-get update && \
    apt-get install -y git ca-certificates && \
    apt-get clean

WORKDIR /workspace/molecular-design
COPY . .

# hack: We need to use the pypi toxsmi package, not the default one
RUN pip uninstall --yes toxsmi && pip install toxsmi && mkdir -p data

# hack: should be done in base gt4sd
RUN pip uninstall --yes torch-scatter torch-sparse torch-cluster torch-geometric && \
    pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html && \
    pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html && \
    pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html && \
    pip install torch-geometric==2.2.0 -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html

RUN chmod +x example_pipeline.sh

ENTRYPOINT ["./example_pipeline.sh"]
