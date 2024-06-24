# Base image containing the installed gt4sd environment
FROM gt4sd-base:latest


# Clone this repository
RUN git clone https://github.com/GT4SD/molecular-design.git

WORKDIR /molecular-design

# We need to use the pypi toxsmi package, not the default one
RUN pip uninstall --yes toxsmi && \
    pip install toxsmi


# Load datasets
RUN mkdir data && \
    python gt4sd-core/scripts/load_data.py \
      --uniprot P05067 \
      --affinity_type IC50 \
      --affinity_cutoff 10000 \
      --output_dir data/ \
      --train_size 0.8 \
      --binary_labels

# Train toxsmi model
RUN python gt4sd-core/scripts/train_toxsmi.py \
      --train data/train.csv \
      --test data/valid.csv \
      --smi data/mols.smi \
      --language tokenizer \
      --params gt4sd-core/config/toxsmi_conf.json \
      --model models \
      --name toxsmi_model

# Generate molecules with MoLeR
RUN head -n 5 data/mols.smi > data/good_docks.smi && \
    python gt4sd-core/scripts/moler_generate_toxsmi.py \
      --smi_path data/good_docks.smi \
      --param_path gt4sd-core/config/moler_conf.json \
      --output_path data/moler_filtered \
      --predictor_path models/toxsmi_model/weights/best_F1.pt

# Generate more diverse molecules with Regression Transformer
RUN python gt4sd-core/scripts/prepare_rt_data.py \
      --smi_path data/moler_filtered/generated.csv \
      --output_path data/moler_filtered/generated_qed.csv && \
    head -n 10 data/moler_filtered/generated_qed.csv > data/moler_filtered/generated_qed_selected.csv && \
    python gt4sd-core/scripts/rt_generate.py \
      --smi_path data/moler_filtered/generated_qed_selected.csv \
      --param_path gt4sd-core/config/rt_conf.json \
      --output_path data/rt


CMD ["bash"]