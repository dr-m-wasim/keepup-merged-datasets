import os
import pandas as pd
from preprocessing.text_cleaner import clean_text
from utils.io_helpers import load_dataset, save_dataset, load_mappings
from utils.df_helpers import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback

DATA_PATHS = {
    "AFND" : "datasets/text datasets/AFND",
    "All-Data": "datasets/text+image/text+image/all_data.csv",
    "Arabic-Satirical-News": "datasets/text datasets/Arabic-Satirical-News/Arabic-Satirical-Fake-News-Dataset-master/fake_news",
    "ANS-Arabic-Satirical-News-train": "datasets/text datasets/ANS-Arabic-Satirical-News/ans-master/data/claim/train.csv",
    "ANS-Arabic-Satirical-News-test": "datasets/text datasets/ANS-Arabic-Satirical-News/ans-master/data/claim/test.csv",
    "ANS-Arabic-Satirical-News-dev": "datasets/text datasets/ANS-Arabic-Satirical-News/ans-master/data/claim/dev.csv",
    "Ax-to-Grind-Urdu-Dataset": "datasets/text datasets/Ax-to-Grind-Urdu/Ax-to-Grind-Urdu-Dataset-main/Combined .csv",
    "BAAI_biendata2019": "datasets/text datasets/BAAI_biendata2019 (10).csv",
    "Badbot": "datasets/text datasets/badbot.csv",
    "BanFakeNews-2.0": "datasets/text datasets/BanFakeNews-2.0.csv",
    "BET-Bend-the-Truth": "datasets/text datasets/BET-Bend-the-Truth/1.Corpus",
    "BuzzFeed-2017": "datasets/text datasets/BuzzFeed-2017/",
    "BuzzFeed-Political-News": "datasets/text datasets/Buzzfeed Political News Dataset",
    "CHECKED-COVID19": "datasets/text datasets/CHECKED-COVID19.csv",
    "Covid-19-Fake-News-Detection": "datasets/text datasets/COVID-19 Fake News Dataset/COVID-19 Fake News Dataset/fake_new_dataset.xlsx",
    "Covid19-FN": "datasets/text datasets/COVID19FN/COVID19FN/COVID19FN.csv",
    "COVID19-Rumor-Data": "datasets/text datasets/COVID19-Rumor-Data/en_dup.csv",
    "CT-FAN-22-eng-test": "datasets/text datasets/CT-FAN-22/FakeNews_Task3_2022/FakeNews_Task3_2022/Task3_Test/English_data_test_release_with_rating.csv",
    "CT-FAN-22-ger-test": "datasets/text datasets/CT-FAN-22/FakeNews_Task3_2022/FakeNews_Task3_2022/Task3_Test/German_data_test_release_with_rating.csv",
    "CT-FAN-22-eng-dev": "datasets/text datasets/CT-FAN-22/FakeNews_Task3_2022/FakeNews_Task3_2022/Task3_train_dev/Task3_english_dev.csv",
    "CT-FAN-22-eng-train": "datasets/text datasets/CT-FAN-22/FakeNews_Task3_2022/FakeNews_Task3_2022/Task3_train_dev/Task3_english_training.csv",
    "English-News": "datasets/text datasets/English-News/fake_or_real_news.csv",
    "Fake.br-Corpus(FNC0, FNC1, FNC2)": "datasets/text+image/text+image/fake.br.corpus.csv",
    "FakeNewsCorpus-Spanish-train": "datasets/text datasets/FakeNewsCorpus-Spanish/FakeNewsCorpusSpanish-master/train.xlsx",
    "FakeNewsCorpus-Spanish-test": "datasets/text datasets/FakeNewsCorpus-Spanish/FakeNewsCorpusSpanish-master/test.xlsx",
    "FakeNewsCorpus-Spanish-dev": "datasets/text datasets/FakeNewsCorpus-Spanish/FakeNewsCorpusSpanish-master/development.xlsx",
    "FakeNewsNet": "datasets/text+image/text+image/Fakenewsnet/FakeNewsNet_Dataset",
    "Fake-News-Urdu": "datasets/text datasets/Fake-News-Urdu/news.xlsx",
    "FA-KES": "datasets/text datasets/FA-KES-Dataset.csv",
    "Fang": "datasets/text datasets/fang/",
    "FANG-Covid": "datasets/text datasets/fang-covid-main/articles",
    "FineFake": "datasets/text datasets/FineFake.pkl",
    "FNID-FakeNewsNet-dev": "datasets/text datasets/FNID-dataset/dataset/fake news detection(FakeNewsNet)/fnn_dev.csv",
    "FNID-FakeNewsNet-train": "datasets/text datasets/FNID-dataset/dataset/fake news detection(FakeNewsNet)/fnn_train.csv",
    "FNID-FakeNewsNet-test": "datasets/text datasets/FNID-dataset/dataset/fake news detection(FakeNewsNet)/fnn_test.csv",
    "FNID-LIAR-dev": "datasets/text datasets/FNID-dataset/dataset/fake news detection(LIAR)/liar_dev.csv",
    "FNID-LIAR-train": "datasets/text datasets/FNID-dataset/dataset/fake news detection(LIAR)/liar_train.csv",
    "FNID-LIAR-test": "datasets/text datasets/FNID-dataset/dataset/fake news detection(LIAR)/liar_test.csv",
    "ISOT": "datasets/text datasets/ISOT",
    "Jruvika-Fake-News-Detection": "datasets/text datasets/Jruvika-Fake-News-Detection/data.csv",
    "Kaggle-Fake-and-Real-News": "datasets/text datasets/Kaggle-Fake-and-Real-News/",
    "Kaggle-Fake-News-test": "datasets/text datasets/Kaggle-Fake-News/test.csv",
    "Kaggle-Fake-News-train": "datasets/text datasets/Kaggle-Fake-News/train.csv/train.csv",
    "Kaggle-Fake-News-Sample": "datasets/text datasets/Kaggle-Fake-News-Sample/resized_v2.csv",
    "Kaggle-Fake-or-Real-News": "datasets/text datasets/Kaggle-Fake-or-Real-News/fake_or_real_news.csv",
    "Kaggle-Getting-Real-about-Fake-news": "datasets/text datasets/Kaggle-Getting-Real-about-Fake-news/fake.csv",
    "LIAR-test": "datasets/text datasets/LIAR/test.tsv", 
    "LIAR-train": "datasets/text datasets/LIAR/train.tsv", 
    "LIAR-valid": "datasets/text datasets/LIAR/valid.tsv",
    "LIAR-PLUS-train" : "datasets/text datasets/LIAR-PLUS/train2.tsv",
    "LIAR-PLUS-test" : "datasets/text datasets/LIAR-PLUS/test2.tsv",
    "LIAR-PLUS-val" : "datasets/text datasets/LIAR-PLUS/val2.tsv",
    "MacIntire": "datasets/text datasets/MacIntire.csv",
    "Misinformation-and-fakenews-and-propaganda": "datasets/text datasets/Misinformation-and-fakenews-and-propaganda/",
    "NewPolitifact": "datasets/text datasets/newpolitifactdataset.csv",
    "Politifact-political-rumor": "datasets/text datasets/Politifact-political-rumor.tsv",
    "Random-Political-News": "datasets/text datasets/Random-Political-News/Public Data/Random Poltical News Dataset",
    "ReCOVery": "datasets/text+image/text+image/ReCOVery/ReCOVery-master/dataset/recovery-news-data.csv",
    "RUN-NewsReliability-train": "datasets/text datasets/RUN-NewsReliability/training_set.json",
    "RUN-NewsReliability-test": "datasets/text datasets/RUN-NewsReliability/test_set.json",
    "Snopes-Claims": "datasets/text datasets/Snopes-claims/Snopes",
    "Snopes-rumors": "datasets/text datasets/Snopes-rumors.tsv",
    "Spanish-Political-News": "datasets/text datasets/spanish-political-fake-news.csv",
    "True-and-Fake-News-election": "datasets/text datasets/True-and-Fake-News/DTN_data/2016_election",
    "True-and-Fake-News-brexit": "datasets/text datasets/True-and-Fake-News/DTN_data/brexit",
    "Truth-seeker-2024": "datasets/text datasets/TruthSeeker2024/TruthSeeker2023/Truth_Seeker_Model_Dataset.csv",
    "UFN-Urdu-Fake-News": "datasets/text datasets/UFN-Urdu-Fake-News/UFN-Urdu-Fake-News/Urdu-Fake-News-master/UFN/UFN",
    "WELFake": "datasets/text datasets/WELFake/WELFake_Dataset.csv"
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
    "Misinformation-and-fakenews-and-propaganda": get_Misinformation_and_fakenews_and_propaganda,
    "Politifact-political-rumor": get_Politifact_political_rumor,
    "Random-Political-News": get_BuzzFeed_Political_News,
    "RUN-NewsReliability-train": get_RUN_NewsReliability,
    "RUN-NewsReliability-test": get_RUN_NewsReliability,
    "Snopes-Claims": get_Snopes_Claims,
    "Snopes-rumors": get_Politifact_political_rumor,
    "Spanish-Political-News": get_Spanish_Political_News,
    "True-and-Fake-News-election": get_True_and_Fake_News,
    "True-and-Fake-News-brexit": get_True_and_Fake_News,
    "UFN-Urdu-Fake-News": get_UFN_Urdu_Fake_News
}

def update_fields(dataset_name, df):
    if dataset_name in ("ANS-Arabic-Satirical-News-train","ANS-Arabic-Satirical-News-test","ANS-Arabic-Satirical-News-dev","Ax-to-Grind-Urdu-Dataset","BAAI_biendata2019", "COVID19-Rumor-Data","Fang","LIAR-test", "LIAR-train", "LIAR-valid", "NewPolitifact", "Politifact-political-rumor","Snopes-rumors","Truth-seeker-2024"):
        df['body'] = ''
    elif dataset_name in ("Arabic-Satirical-News", "BET-Bend-the-Truth","CHECKED-COVID19","Fake.br-Corpus(FNC0, FNC1, FNC2)","Fake-News-Urdu","FineFake","Misinformation-and-fakenews-and-propaganda", "True-and-Fake-News-election", "True-and-Fake-News-brexit", "UFN-Urdu-Fake-News"):
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