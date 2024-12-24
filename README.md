# Machine Learning Approaches to the Shafarevich-Tate Group of Elliptic Curves

This is the codebase for the [paper](link) of the same name, by Angelica Babei, Barinder S. Banwait, AJ Fong, Xiaoyu Huang, and Deependra Singh.

## How do I reproduce the results in your paper?

You will first need the datafiles. These must be placed in a subdirectory called "sha" in the `data_files` folder. The data we have used for this project has been obtained from the LMFDB. For the reader's convenience, it may be downloaded from [here](https://www.dropbox.com/scl/fo/gumbemoahrec5opot4nyo/AJt1wONoqi1fz850ncOolLY?rlkey=ply0vu4tfmq43tojnwnpy8mgm&e=3&st=hxn4h7i7&dl=0) and [here](https://drive.google.com/file/d/1XzcpjAoWE-EPbcgUOaWpvix11JgQDLl9/view?usp=sharing) (both are used in the notebooks, but this latter one is the more relevant one). We cannot have this in the github repo because it is far too large. The reader is encouraged to download these files and place them in the `sha` subdirectory of `data_files`.

Once you've got the data files there, you will find all relevant code in the various `.ipynb` files in the `experiments` folder.

## Directory structure

```
.
├── ml_rnt              # Main project directory
│   ├── data_files      # Where the data is stored
│   ├── python          # Python scripts for data processing and modeling
│   ├── experiments     # Jupyter notebooks for experiments and analyses
│   ├── trained_models  # Used for storing large NN models that take time to train
│   ├── LICENSE         # License for the project
│   ├── README.md       # Project description and setup instructions
│   ├── .gitignore      # Git ignore file for unwanted files
```

# Copyright

    ####  Copyright (C) 2024 The Authors: Angelica Babei, Barinder S. Banwait,
    AJ Fong, Xiaoyu Huang, and Deependra Singh.

    The code found throughout this repository  is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    The authors can be reached at the email addresses stated in the paper
    linked to above.