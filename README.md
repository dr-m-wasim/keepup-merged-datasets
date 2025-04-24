# keepup-merged-datasets

This repository provides a unified and harmonized version of **68 fake news detection datasets**, each containing the fields: `title`, `body`, and `label`. The final dataset follows a **binary classification** format with labels `fake` and `real`.

## Main Script

- **`merge_datasets.py`**  
  This is the main script of the repository. It defines the **paths to all 68 datasets** and performs the merging process. It utilizes helper modules for reading, cleaning, and aligning data according to the label and feature mappings.

## Folder Structure

### `configs/`
- **`mappings.json`**  
  Contains all the necessary **label mappings** (e.g., which original labels map to `fake` or `real`) and **feature mappings** for aligning dataset columns (e.g., mapping dataset-specific column names to `title`, `body`, and `label`).

### `preprocessing/`
- **`text_cleaner.py`**  
  Contains functions to **clean and normalize text**, including:
  - Removing URLs
  - Removing punctuation
  - Normalizing white spaces

### `utils/`
- **`io_helpers.py`**  
  Utility functions to **read various file formats**, including:
  - CSV
  - TSV
  - JSON
  - XLSX

- **`df_helpers.py`**  
  Helper functions for **working with and cleaning pandas DataFrames**, particularly for reading and preparing datasets for merging.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/dr-m-wasim/keepup-merged-datasets.git
   cd keepup-merged-datasets

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the merge script:
   ```bash
   python merge_datasets.py
