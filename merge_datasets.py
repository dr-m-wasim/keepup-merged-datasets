import os
import pandas as pd
from preprocessing.text_cleaner import clean_text
from utils.io_helpers import load_dataset, save_dataset, load_mappings
from utils.df_helpers import get_AFND_dataframe
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback

DATA_PATHS = {
    #"AFND" : "datasets/text datasets/AFND",
    "All-Data": "datasets/text+image/text+image/all_data.csv",
    "Badbot" : "datasets/text datasets/badbot.csv"
}

# fill this dictionary for only special cases
convert_to_df = {
    "AFND" : get_AFND_dataframe
}


def process_and_map(dataset_name, df, mapping):
    # Extract the column mappings for current dataset
    column_mapping = mapping['column_mapping']
    label_mapping = mapping['label_mapping']

    # Rename columns to standard names
    df = df[[column_mapping["title"], column_mapping["body"], column_mapping["label"]]].dropna()
    df = df.rename(columns={
        column_mapping["title"]: "title",
        column_mapping["body"]: "body",
        column_mapping["label"]: "label"
    })

    labels_to_keep = list(label_mapping.keys())
    
    if len(labels_to_keep) != 0:
        
        # Convert the labels to string
        df['label'] = df['label'].astype(str)
        
        # Only select the rows which contain the specific labels defined in the mapping values
        df = df[df['label'].isin(labels_to_keep)]

        # Map credible to real and not credible to fake
        df['label'] = df['label'].map(label_mapping)
        
    # Clean text fields
    df["title"] = df["title"].astype(str).apply(clean_text)
    df["body"] = df["body"].astype(str).apply(clean_text)

    return df[["title", "body", "label"]]


def main():

    mappings = load_mappings("configs/mappings.json")

    merged = []

    def process_dataset(datasert_name, path):
        if datasert_name in convert_to_df.keys():
            df = convert_to_df[datasert_name](path)
        else:
            df = load_dataset(path)
        return process_and_map(datasert_name, df, mappings[datasert_name])

     # Submit tasks to ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = {executor.submit(process_dataset, name, path): name for name, path in DATA_PATHS.items()}
        
        # Show progress with tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing datasets"):
            try:
                result_df = future.result()
                merged.append(result_df)
            except Exception as e:
                name = futures[future]
                print(f"Error processing {name}: {e}")
                traceback.print_exc()

    
    # Concatenate all processed DataFrames
    final_df = pd.concat(merged, ignore_index=True)
    os.makedirs("processed_data", exist_ok=True)
    save_dataset(final_df, "processed_data/keepup-multilingual.parquet")
    print(f"Merged dataset saved with {len(final_df)} records and columns: {list(final_df.columns)}")

if __name__ == '__main__':
    main()