# GAPNN
PathNN is a GNN model that updates a node's representation by aggregating path information from other nodes in the graph. Unlike standard GNNs, which focus only on a node's direct neighbors, PathNN considers all nodes connected through paths, enabling it to capture richer structural information. The core principle of PathNN is to enhance node representations using paths. Each node’s feature vector is updated by computing and aggregating information from paths of different lengths connecting to that node.

## Installation

### Requirements
- conda: 4.6+
- python 3.9x
- PyTorch 1.12.1 https://pytorch.org/get-started/previous-versions/#linux-and-windows-7
- Cuda 10.2

### Setting up virtual environment

#### Create a conda virtual environment
```bash
conda env create -f environment.yml
conda activate GAPNN
```

## Run Experiments

```bash
python example.py
```
### Download the CHM13 reference
The CHM13 reference will be download and set up the directory structure for simulating reads, constructing graphs and running experiments.
If failed, you can download the CHM13 by "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/chm13.draft_v1.1.fasta.gz"

### Download the real data
You can download the real dataset separately by accessing the link in download_dataset.sh:
```bash
https://www.dropbox.com/s/fa14gza4cf9dsk3/genomic_dataset_chunk.z01?dl=1
https://www.dropbox.com/s/i8pftsjmbpkj1a0/genomic_dataset_chunk.z02?dl=1
https://www.dropbox.com/s/udlqbypizummctq/genomic_dataset_chunk.z03?dl=1
https://www.dropbox.com/s/2qzbswupfg90tbq/genomic_dataset_chunk.z04?dl=1
https://www.dropbox.com/s/0suo9k6fhtdg4p3/genomic_dataset_chunk.zip?dl=1
```

Save zip file in `<data_path>/real/` After download you need to run the bash
```bash
zip --fix genomic_dataset_chunk --out genomic_dataset
unzip genomic_dataset.zip
```

### Specify the train/valid/test split
Editing `_train_dict`, `valid_dict`, and `_test_dict` inside `config.py`.
```python
_train_dict = {'chr19': 2, 'chr20': 1}
_valid_dict = {'chr19': 1}
_test_dict = {'chr21_r': 1}
```

### Run the pipeline
```bash
python pipeline.py --data <data_path> --out <out_name>
```
