# Base image containing the installed gt4sd environment
#FROM drugilsberg/gt4sd-base:v1.4.2-cpu
FROM quay.io/gt4sd/gt4sd-base:v1.4.2-cpu


# Certs for git clone
RUN apt-get update && \
    apt-get install -y git ca-certificates && \
    apt-get clean

RUN git clone https://github.com/GT4SD/molecular-design.git
WORKDIR /workspace/molecular-design

# hack: We need to use the pypi toxsmi package, not the default one
RUN pip uninstall --yes toxsmi && pip install toxsmi && mkdir data

# hack: should be done in base gt4sd
RUN pip uninstall --yes torch-scatter torch-sparse torch-cluster torch-geometric && \
    pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html && \
    pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html && \ 
    pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html && \
    pip install torch-geometric -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html

RUN chmod +x example_pipeline.sh

ENTRYPOINT ["./example_pipeline.sh"]
