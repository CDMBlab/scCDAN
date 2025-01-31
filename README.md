# scCDAN:constraint domain adaptation network for cell type annotation across single cell RNA sequening data

In this study, a constraint domain adaptation network, named
scCDAN, is proposed to address the problem of batch effects that lead to
poor cell type annotation. Firstly, a domain alignment module is used to
align the distributions of source and target domain data through adversarial
learning strategies to reduce domain discrepancies. Furthermore, to achieve
finer granularity in differentiating cell types, a category boundary constraint
module is designed to regulate the positional relationships between cells of
the same type and those of different types in the feature space. scCDAN
combines domain alignment and category boundary constraint modules to
directly and indirectly address batch effect issues, thereby improving the
accuracy of cell type annotation. The effectiveness of scCDAN is validated on
simulated, cross-platforms, and cross-species datasets. Experimental results
demonstrate that scCDAN outperforms comparative methods in both cell
type annotation and batch correction.

# Requirements
### Enviroment
- pyTorch >= 1.1.0  
- scanpy >= 1.9.1  
- python >=3.7.0 
- umap-learn>=0.5.2
- numpy >=1.26.4
- umap-learn >=0.5.3
- matplotlib >=3.7.1
- pandas >=2.0.1
- scikit-learn>=0.21.1
- scipy>=1.3.0
- universal-divergence>=0.2.0

### Installation

For a traditional Python installation, use
```bash
 pip install "$package name"
```
Users can install all dependencies with the following command:
```bash
pip install -r requirements.txt
```
Users can install all dependencies in the GitHub path:
```bash
pip install git+https://github.com/CDMBlab/scCDAN.git

```
# Data preparation

Log-normalized count matrix is recommonded as the input of scCDAN. Raw counts matrix can be normalized by the NormalizeData function in Seurat with default ‘LogNormalize’ normalization method and a scale factor of 10,000. We use top 2000 highly variable genes as input features. We provided an example of processed datasets in the `scCDAN/processed_data` folder as follows:
`combine_expression.csv` is the expression matrix [gene, cell] combining source and target.
`combine_labels.csv` is the cell type label array combining source and target.
`domain_labels.csv` is the batch/domain label array combining source and target.
`digit_label_dict.csv` is the one-to-one mapping between digital label and cell type label.

# Additional data input options
### H5
```bash
import anndata
import numpy as np
import scipy

def read_data(dataname):
    adata = anndata.read(dataname)
    mat= adata.X
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    obs = adata.obs
    cell_name = np.array(obs["cell_ontology_class"])
    if (cell_name == "").sum() > 0:
        cell_name[cell_name == ""] = "unknown_class"
    return X, cell_name
```
### H5ad
```bash
import h5py
import numpy as np

def read_data(datath):
    data_mat = h5py.File(datath)
    X = np.array(data_mat["X"])
    cell_name = np.array(data_mat["Y"])
    batch = np.array(data_mat["B"])
    return X, cell_name, batch
```

# Usage

### Python script usage

An example of how to use scCDAN for both classification and batch correction is:

```bash
python main.py --dataset_path path/to/input/files
               --result_path path/for/output/files
               --source_name batch name
               --target_name batch name
	       --gpu_id GPU id to run
```

The `dataset_path` must contain the four CSV files preprocessed as in `scCDAN/processed_data` folder. In `result_path`, there will be three output files: `final_model_*.ckpt` has the trained model    parameters (i.e. weights and biases) and can be loaded for label prediction. `pred_labels_*.csv` contains the predicted cell label and corresponding confidence score (softmax probability). `embeddings_*.csv` contains the batch-corrected low-dimensional embeddings (default is 256) for visualization.


# Questions

For questions about the datasets and codes, please contact [saltsalt412@163.com].

