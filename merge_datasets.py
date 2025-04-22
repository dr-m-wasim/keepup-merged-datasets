import os
import pandas as pd
from preprocessing.text_cleaner import clean_text
from utils.io_helpers import load_dataset, save_dataset, load_mappings
from utils.df_helpers import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback

DATA_PATHS = {
    #"AFND" : "datasets/text datasets/AFND",
    "All-Data": r'D:\text+image\text+image\all_data.csv',
    #"Arabic-Satirical-News": r'D:\text datasets\text datasets\Arabic-satirical-news',
    #"ANS-Arabic-Satirical-News-train": r'D:\text datasets\text datasets\ans-master\ans-master\data\claim\train.csv',
    #"ANS-Arabic-Satirical-News-test": r'D:\text datasets\text datasets\ans-master\ans-master\data\claim\test.csv',
    #"ANS-Arabic-Satirical-News-dev": r'D:\text datasets\text datasets\ans-master\ans-master\data\claim\dev.csv',
    #"Ax-to-Grind-Urdu-Dataset": r'D:\text datasets\text datasets\Ax-to-Grind-Urdu-Dataset-main\Ax-to-Grind-Urdu-Dataset-main\Combined .csv',
    #"BAAI_biendata2019": r'D:\text datasets\text datasets\BAAI_biendata2019.csv',
    #"Badbot": r'D:\text datasets\text datasets\badbot.csv',
    #"BanFakeNews-2.0": r'D:\text datasets\text datasets\BanFakeNews-2.0.csv',
    #"BET-Bend-the-Truth": r'D:\text datasets\text datasets\BET-Bend-the-Truth\Datasets-for-Urdu-news-master\Urdu Fake News Dataset\1.Corpus',
    #"BuzzFeed-2017": r'D:\text datasets\text datasets\BuzzFeed-2017',
    #"BuzzFeed-Political-News": r'D:\text datasets\text datasets\Buzzfeed Political News Dataset',
    #"CHECKED-COVID19": r'D:\text datasets\text datasets\CHECKED-COVID19.csv',
    #"Covid-19-Fake-News-Detection": r'D:\text datasets\text datasets\COVID-19 Fake News Dataset\COVID-19 Fake News Dataset\fake_new_dataset.csv'
    #"Covid19-FN": r'D:\text datasets\text datasets\COVID19FN\COVID19FN\COVID19FN.csv',
    #"COVID19-Rumor-Data": r'D:\text datasets\text datasets\COVID19-Rumor-Data\en_dup.csv',
    # "CT-FAN-22-eng-test": r'D:\text datasets\text datasets\CT-FAN-22\FakeNews_Task3_2022\Task3_Test\English_data_test_release_with_rating.csv',
    # "CT-FAN-22-ger-test": r'D:\text datasets\text datasets\CT-FAN-22\FakeNews_Task3_2022\Task3_Test\German_data_test_release_with_rating.csv',
    # "CT-FAN-22-eng-dev": r'D:\text datasets\text datasets\CT-FAN-22\FakeNews_Task3_2022\Task3_train_dev\Task3_english_dev.csv',
    # "CT-FAN-22-eng-train": r'D:\text datasets\text datasets\CT-FAN-22\FakeNews_Task3_2022\Task3_train_dev\Task3_english_training.csv',
    #"English-News": r'D:\text datasets\text datasets\English-News\fake_or_real_news.csv',
    #"Fake.br-Corpus(FNC0, FNC1, FNC2)": r'D:\text+image\text+image\fake.br.corpus.csv',
    #"FakeNewsCorpus-Spanish-train": r'D:\text datasets\text datasets\FakeNewsCorpusSpanish\FakeNewsCorpusSpanish-master\train.xlsx',
    #"FakeNewsCorpus-Spanish-test": r'D:\text datasets\text datasets\FakeNewsCorpusSpanish\FakeNewsCorpusSpanish-master\test.xlsx',
    #"FakeNewsCorpus-Spanish-dev": r'D:\text datasets\text datasets\FakeNewsCorpusSpanish\FakeNewsCorpusSpanish-master\development.xlsx',
    #"FakeNewsNet": r'D:\text+image\text+image\Fakenewsnet\FakeNewsNet_Dataset\FakeNewsNet_Dataset',
    #"Fake-News-Urdu": r'D:\text datasets\text datasets\Fake-News-Urdu.xlsx',
    #"FA-KES": r'D:\text datasets\text datasets\FA-KES-Dataset.csv',
    #"Fang": r'D:\text datasets\text datasets\fang',
    #"FANG-Covid": r'D:\text datasets\text datasets\fang-covid-main\fang-covid-main\articles',
    #"FineFake": r'D:\text datasets\text datasets\FineFake.pkl',
    #"FNID-FakeNewsNet-dev": r'D:\text datasets\text datasets\fake news detection(FakeNewsNet)\fnn_dev.csv',
    #"FNID-FakeNewsNet-train": r'D:\text datasets\text datasets\fake news detection(FakeNewsNet)\fnn_train.csv',
    #"FNID-FakeNewsNet-test": r'D:\text datasets\text datasets\fake news detection(FakeNewsNet)\fnn_test.csv',
    #"FNID-LIAR-dev": r'D:\text datasets\text datasets\fake news detection(LIAR)\liar_dev.csv',
    #"FNID-LIAR-train": r'D:\text datasets\text datasets\fake news detection(LIAR)\liar_train.csv',
    #"FNID-LIAR-test": r'D:\text datasets\text datasets\fake news detection(LIAR)\liar_test.csv',
    #"ISOT": r'D:\text datasets\text datasets\ISOT',
    #"Jruvika-Fake-News-Detection": r'D:\text datasets\text datasets\Jruvika-Fake-News-Detection\data.csv',
    #"Kaggle-Fake-and-Real-News": r'D:\text datasets\text datasets\Kaggle-Fake-and-Real-News',
    #"Kaggle-Fake-News-test": r'D:\text datasets\text datasets\Kaggle-Fake-News\test',
    #"Kaggle-Fake-News-train": r'D:\text datasets\text datasets\Kaggle-Fake-News\train\train.csv',
    #"Kaggle-Fake-News-Sample": r'D:\text datasets\text datasets\Kaggle-Fake-News-Sample\resized_v2.csv',
    #"Kaggle-Fake-or-Real-News": r'D:\text datasets\text datasets\Kaggle-Fake-or-Real-News\fake_or_real_news.csv',
    #"Kaggle-Getting-Real-about-Fake-news": r'D:\text datasets\text datasets\Kaggle-Getting-Real-about-Fake-news\fake.csv',
    #"LIAR-test": r'D:\text datasets\text datasets\LIAR\test.tsv', 
    #"LIAR-train": r'D:\text datasets\text datasets\LIAR\train.tsv', 
    #"LIAR-valid": r'D:\text datasets\text datasets\LIAR\valid.tsv',
    #"LIAR-PLUS-train" : r'D:\text datasets\text datasets\LIAR-PLUS\train2.tsv',
    #"LIAR-PLUS-test" : r'D:\text datasets\text datasets\LIAR-PLUS\test2.tsv',
    #"LIAR-PLUS-val" : r'D:\text datasets\text datasets\LIAR-PLUS\val2.tsv',
    #"MacIntire": r'D:\text datasets\text datasets\MacIntire.csv',
    "Misinformation-and-fakenews-and-propaganda": r'D:\text datasets\text datasets\Misinformation-and-fakenews-and-propaganda',
}

# fill this dictionary for only special cases
convert_to_df = {
    "AFND" : get_AFND_dataframe,
    "Arabic-Satirical-News": get_Arabic_Satirical_News,
    "BET-Bend-the-Truth": get_BET_Bend_the_Truth,
    "BuzzFeed-2017": get_BuzzFeed_2017,
    "BuzzFeed-Political-News": get_BuzzFeed_Political_News,
    "FakeNewsCorpus-Spanish-train": get_excel_datasets,
    "FakeNewsCorpus-Spanish-test": get_excel_datasets,
    "FakeNewsCorpus-Spanish-dev": get_excel_datasets,
    "FakeNewsNet": get_FakeNewsNet,
    "Fake-News-Urdu": get_excel_datasets,
    "FA-KES": get_FA_KES,
    "Fang": get_fang,
    "FANG-Covid": get_FANG_Covid,
    "FineFake": get_FineFake,
    "ISOT": get_ISOT,
    "Kaggle-Fake-and-Real-News": get_Kaggle_Fake_and_Real_News,
    "Kaggle-Fake-News-test": get_Kaggle_Fake_News_test,
    "LIAR-test": get_LIAR,
    "LIAR-train": get_LIAR,
    "LIAR-valid": get_LIAR,
    "LIAR-PLUS-train": get_LIAR_PLUS,
    "LIAR-PLUS-test": get_LIAR_PLUS,
    "LIAR-PLUS-val": get_LIAR_PLUS,
    "Misinformation-and-fakenews-and-propaganda": get_Misinformation_and_fakenews_and_propaganda
}

def update_fields(dataset_name, df):
    if dataset_name in ("ANS-Arabic-Satirical-News-train","ANS-Arabic-Satirical-News-test","ANS-Arabic-Satirical-News-dev","Ax-to-Grind-Urdu-Dataset","BAAI_biendata2019", "COVID19-Rumor-Data","Fang","LIAR-test", "LIAR-train", "LIAR-valid"):
        df['body'] = ''
    elif dataset_name in ("Arabic-Satirical-News", "BET-Bend-the-Truth","CHECKED-COVID19","Fake.br-Corpus(FNC0, FNC1, FNC2)","Fake-News-Urdu","FineFake","Misinformation-and-fakenews-and-propaganda"):
        df['title'] = ''

    return df

def process_and_map(dataset_name, df, mapping):
    # Extract the column mappings for current dataset
    column_mapping = mapping['column_mapping']
    label_mapping = mapping['label_mapping']

    df = update_fields(dataset_name, df)

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
        df['label'] = df['label'].astype(str).str.strip().map(label_mapping)

    # Clean text fields
    df["title"] = df["title"].astype(str).apply(clean_text)
    df["body"] = df["body"].astype(str).apply(clean_text)

    return df[["title", "body", "label"]]


def main():

    mappings = load_mappings("configs/mappings.json")

    merged = []

    
    def process_dataset(dataset_name, path):
        if dataset_name in convert_to_df.keys():
            df = convert_to_df[dataset_name](path)
        else:
            df = load_dataset(path)
        return process_and_map(dataset_name, df, mappings[dataset_name])
    

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