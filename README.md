
# **Lorem Ipsum**

### Overview

Official Python implementation for another random work
    

### Repository Structure

```
FairLinBandit/
├── algorithms/       # Contains code for all bandit algorithms used
├── cached/           # Stores regret data from first-time runs
├── plots/            # Final figures
├── main.py           # Master script: runs all experiments sequentially
├── experiments.py    # Functions for running and plotting individual experiments
├── data_loader.py    # Functions for loading datasets
└── requirements.txt  # Python dependencies

```


    

#### `cached/`

Stores serialized regret data generated during the first run of each experiment which can be directly loaded for subsequent runs.

#### `plots/`

Holds the final plot images (e.g., `.png` files) for each experiment. These plots compare algorithmic performance under different settings.

### Installation

1.  **Clone the repository**:
    
    ```bash
    git clone https://github.com/NP-Hardest/FairLinBandit.git
    cd FairLinBandit
    
    ```
    
2.  **Create a conda environment** (optional but recommended):
    
    ```bash
    conda create -n FairLinBandit 
    conda activate FairLinBandit
    
    ```
    One can also use venv instead of conda.
    
3.  **Install dependencies**:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    

### Usage

#### Run All Experiments

Execute the master script to run all experiments in sequence. Use `np.random.seed(42)` for reproducing results from the paper:

```bash
python main.py

```

#### Tweak Parameters

 Adjust experiment parameters in  `main.py` such as:

    -   Horizon length (T)
        
    -   Number of trials
        
    -   Algorithm-specific settings

    -   $p$ and $d$ values.
        

While running the experiments for the first time, the regret data is stored in `cached/` directory. For subsequent runs, this data can be directly loaded using `np.load()`.


#### Results & Visualization

-   After running the experiments, plots will be saved in the `plots/` folder.
    
-   Use the functions in `experiments.py` to customize plot styles or export formats.
    



#### Contributing

Feel free to fork the repository to reproduce our results and other suggestions for modifications.
    
