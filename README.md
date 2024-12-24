# Machine Learning Approaches to the Shafarevich-Tate Group of Elliptic Curves

This is the codebase for the [paper](link) of the same name, by Angelica Babei, Barinder S. Banwait, AJ Fong, Xiaoyu Huang, and Deependra Singh.

## How do I reproduce the results in your paper?

You will first need the datafiles. These must be placed in a subdirectory called "sha" in the `data_files` folder. The data we have used for this project has been obtained from the LMFDB. For the reader's convenience, it may be downloaded from [here](link). We cannot have this in the github repo because it is far too large.

Once you've got the data files there, you will find all relevant code in the various `.ipynb` files in the `experiments` folder.

## Directory structure

```
.
├── ml_rnt              # Main project directory
│   ├── data_files      # Where the data is stored
│   ├── python          # Python scripts for data processing and modeling
│   ├── experiments      # Jupyter notebooks for experiments and analyses
│   ├── LICENSE         # License for the project
│   ├── README.md       # Project description and setup instructions
│   ├── .gitignore      # Git ignore file for unwanted files
│   └── .DS_Store       # System metadata (can be ignored or removed)
```