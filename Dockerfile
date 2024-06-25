# Base image containing the installed gt4sd environment
FROM first_test_img:latest


# Certs for git clone
RUN apt-get update && \
    apt-get install -y git ca-certificates && \
    apt-get clean
# Clone this repository
RUN git clone https://github.com/GT4SD/molecular-design.git

WORKDIR /molecular-design

# We need to use the pypi toxsmi package, not the default one
RUN pip uninstall --yes toxsmi && \
    pip install toxsmi


# Load datasets
RUN mkdir data && \
    python scripts/load_data.py \
      --uniprot P05067 \
      --affinity_type IC50 \
      --affinity_cutoff 10000 \
      --output_dir data/ \
      --train_size 0.8 \
      --binary_labels

# Train toxsmi model
RUN python scripts/train_toxsmi.py \
      --train data/train.csv \
      --test data/valid.csv \
      --smi data/mols.smi \
      --language tokenizer \
      --params config/toxsmi_conf.json \
      --model models \
      --name toxsmi_model

# Generate molecules with MoLeR
RUN head -n 5 data/mols.smi > data/good_docks.smi && \
    python scripts/moler_generate_toxsmi.py \
      --smi_path data/good_docks.smi \
      --param_path config/moler_conf.json \
      --output_path data/moler_filtered \
      --predictor_path models/toxsmi_model/weights/best_F1.pt

# Generate more diverse molecules with Regression Transformer
RUN python scripts/prepare_rt_data.py \
      --smi_path data/moler_filtered/generated.csv \
      --output_path data/moler_filtered/generated_qed.csv && \
    head -n 10 data/moler_filtered/generated_qed.csv > data/moler_filtered/generated_qed_selected.csv && \
    python scripts/rt_generate.py \
      --smi_path data/moler_filtered/generated_qed_selected.csv \
      --param_path config/rt_conf.json \
      --output_path data/rt


CMD ["bash"]
