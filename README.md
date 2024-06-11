 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# AI For Target Based Drug Design


### Install [GT4SD](https://github.com/GT4SD/gt4sd-core) 
```bash
git clone https://github.com/GT4SD/gt4sd-core.git
cd gt4sd-core/
conda env create -f conda_gpu.yml
conda activate gt4sd
pip install gt4sd
```

Uninstall and reinstall toxsmi using `pip` to get correct version:
```bash
pip uninstall toxsmi
pip install toxsmi
```

### (Optional) Download data from [BindingDB](https://www.bindingdb.org/bind/BindingDBRESTfulAPI.jsp)
Example of retrieving binding data for UniProt target P05067 (kinase):
```bash
python scripts/load_data.py \
    --uniprot P05067 \
    --affinity_type IC50 \
    --affinity_cutoff 10000 \
    --output_dir data/ \
    --train_size 0.8 \
    --binary_labels
```

### Train the screening model [Toxsmi](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00099g)

Assuming the data sets reside in the `data` folder either by running the step above or symlinking your own datasets,
you can start the training with the following command:
```
python scripts/train_toxsmi.py \
    --train data/train.csv \
    --test data/valid.csv \
    --smi data/mols.smi \
    --language tokenizer \
    --params config/toxsmi_conf.json \
    --model models \
    --name toxsmi_model
```

Read the Toxsmi paper for more details: [link](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00099g)

### Generate molecules with [MoLeR](https://github.com/microsoft/molecule-generation)
Here is an example of taking the first molecule
```bash
head -n 5 data/mols.smi > data/good_docks.smi 
```

After training the screening model, we generate molecules using the following command:
```
python scripts/moler_generate_toxsmi.py \
    --smi_path data/good_docks.smi \
    --param_path config/moler_conf.json \
    --output_path data/moler_filtered \
    --predictor_path models/toxsmi_model/weights/best_F1.pt
```
where `best_F1.pt` is the weights of the best toxsmi model.

Read the MoLeR paper for more details: [link](https://arxiv.org/abs/2103.03864)

### Generate more diverse molecules with [Regression Transformer](https://www.nature.com/articles/s42256-023-00639-z)
Generate the dataset
```bash
python scripts/prepare_rt_data.py \
    --smi_path data/moler_filtered/generated.csv \
    --output_path data/moler_filtered/generated_qed.csv 
```
```
python scripts/rt_generate.py \
    --smi_path data/moler_filtered/generated_qed.csv \
    --param_path config/rt_conf.json \
    --output_path data/rt
```
Read the Regression Transformer paper for more details: [link](https://www.nature.com/articles/s42256-023-00639-z)

### Run inference on [Toxsmi](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00099g)
After generating a more diverse set of molecules, we screen the newly generated molecules a final time using toxsmi.
First we structure the input dataset by running:
```
python scripts/inference_dataset.py -i data/rt/qed_rt_conf_generated_qed/generated.csv
```
This generates `dummy_data.csv` and `generated.smi`. Run the inference:
```
python scripts/test_toxsmi.py \
    --model_path models/toxsmi_model \
    --smi_filepath generated.smi \
    --label_filepath dummy_data.csv \
    --checkpoint_name F1
```
this results in `models/toxsmi_model/results/dummy_data_F1_results_flat.csv` which contain the predictions.

### Computing properties with GT4SD
```bash
python scripts/mol_properties.py \
    --smi_path models/toxsmi_model/results/dummy_data_F1_results_flat.csv \
    --output_path mol_props.csv 
```

### Retrosynthesis with [RXN](https://rxn.app.accelerate.science/)
```bash
pip install rxn4chemistry
```

A free API key can be generated at [RXN](https://rxn.app.accelerate.science/) by creating an account.
To run the retrosynthesis a project id is also needed, this can be extracted from the url, which may look like this
`https://rxn.app.accelerate.science/rxn/projects/<project id is here>/test/dashboard`.

Since retrosynthesis is time consuming, it is recommended to rank your molecules and only retrosynthesize the top n values.
Here is an example of taking the first molecule
```bash
head -n 2 data/rt/qed_rt_conf_generated_qed/generated.csv > selected_for_retro.csv
```

```bash
API_KEY=<your API key here>
python scripts/retrosynthesis.py selected_for_retro.csv \
--api_key $API_KEY \
--project_id <your project id here> \
--steps 4 \
--timeout 100 \
--name dummy_name
```

For further information on RXN refer to the papers:
- [https://pubs.rsc.org/en/content/articlehtml/2020/sc/c9sc05704h](https://pubs.rsc.org/en/content/articlehtml/2020/sc/c9sc05704h)
- [https://www.nature.com/articles/s41524-024-01290-x](https://www.nature.com/articles/s41524-024-01290-x)