python scripts/load_data.py \
      --uniprot P05067 \
      --affinity_type IC50 \
      --affinity_cutoff 10000 \
      --output_dir data/ \
      --train_size 0.8 \
      --binary_labels

# Train toxsmi model
python scripts/train_toxsmi.py \
      --train data/train.csv \
      --test data/valid.csv \
      --smi data/mols.smi \
      --language tokenizer \
      --params config/toxsmi_conf.json \
      --model models \
      --name toxsmi_model

# Generate molecules with MoLeR
head -n 5 data/mols.smi > data/good_docks.smi
python scripts/moler_generate_toxsmi.py \
      --smi_path data/good_docks.smi \
      --param_path config/moler_conf.json \
      --output_path data/moler_filtered \
      --predictor_path models/toxsmi_model/weights/best_F1.pt

# Generate more diverse molecules with Regression Transformer
python scripts/prepare_rt_data.py \
      --smi_path data/moler_filtered/generated.csv \
      --output_path data/moler_filtered/generated_qed.csv && \
head -n 10 data/moler_filtered/generated_qed.csv > data/moler_filtered/generated_qed_selected.csv
python scripts/rt_generate.py \
      --smi_path data/moler_filtered/generated_qed_selected.csv \
      --param_path config/rt_conf.json \
      --output_path data/rt

python scripts/inference_dataset.py -i data/rt/qed_rt_conf_generated_qed_selected/generated.csv

python scripts/test_toxsmi.py \
    --model_path models/toxsmi_model \
    --smi_filepath generated.smi \
    --label_filepath dummy_data.csv \
    --checkpoint_name F1

# Calculate properties
python scripts/mol_properties.py \
    --smi_path models/toxsmi_model/results/dummy_data_F1_results_flat.csv \
    --output_path mol_props.csv 

# Run RXN 
pip install rxn4chemistry
head -n 2 data/rt/qed_rt_conf_generated_qed_selected/generated.csv > selected_for_retro.csv
python scripts/retrosynthesis.py selected_for_retro.csv \
--api_key $API_KEY \
--project_id $PROJ_ID \
--steps 4 \
--timeout 100 \
--name my_retrosynthesis