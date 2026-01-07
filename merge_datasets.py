import os
import pandas as pd
from preprocessing.text_cleaner import clean_text
from utils.io_helpers import load_dataset, save_dataset, load_mappings
from utils.df_helpers import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback

DATA_PATHS = {
#      "AFND" : ["datasets/text datasets/AFND", "ar", "others", "misbar"],
#       "All-Data": ["datasets/text+image/text+image/all_data.csv", "en", "politics,sports", "new-york-times,washington-post"],
#     "Arabic-Satirical-News": ["datasets/text datasets/Arabic-Satirical-News/Arabic-Satirical-Fake-News-Dataset-master/fake_news", "ar", "others", "al-hudood,al-ahram,al-mexici"],
#     "ANS-Arabic-Satirical-News-train": ["datasets/text datasets/ANS-Arabic-Satirical-News/ans-master/data/claim/train.csv", "ar", "politics,sports,entertainment,health", "N\A"],
#     "ANS-Arabic-Satirical-News-test": ["datasets/text datasets/ANS-Arabic-Satirical-News/ans-master/data/claim/test.csv", "ar", "politics,sports,entertainment,health", "N\A"],
#     "ANS-Arabic-Satirical-News-dev": ["datasets/text datasets/ANS-Arabic-Satirical-News/ans-master/data/claim/dev.csv", "ar", "politics,sports,entertainment,health", "N\A"],
#     "Ax-to-Grind-Urdu-Dataset": ["datasets/text datasets/Ax-to-Grind-Urdu/Ax-to-Grind-Urdu-Dataset-main/Combined .csv", "ur", "politics,health,sports,entertainment,technology,weather,agriculture,economy,showbiz,social-media,education,womens-rights,religion,foreign-affairs,international", "jang,dawn-news-urdu,express-roaznama,geo-urdu,bbc-urdu,indian-etemaad,inquilab,hindustan-express,siasat"],
#     "BAAI_biendata2019": ["datasets/text datasets/BAAI_biendata2019 (10).csv", "zh", "health", "baai"],
#     "Badbot": ["datasets/text datasets/badbot.csv", "en", "politics", "N\A"],
#     "BanFakeNews-2.0": ["datasets/text datasets/BanFakeNews-2.0.csv", "bn", "politics,miscellaneous,international,lifestyle,medical,religious,sports,educational,technology,national,crime,entertainment,finance", "N\A"],
#     "BET-Bend-the-Truth": ["datasets/text datasets/BET-Bend-the-Truth/1.Corpus", "ur", "technology,education,business,sports,politics,entertainment", "bbcnews,cnnurdu,dawnnews,dailypakistan,eteemadnews,express-news,hamariweb,jung-news,mashriq-news,nawaiwaqt-news,roznama-dunya,the-daily-siasat,urdu-news-room,urdupoint,voice-of-america,waqt-news"],
#     "BuzzFeed-2017": ["datasets/text datasets/BuzzFeed-2017/", "en", "politics", "9-news-publishers"],
#     "BuzzFeed-Political-News": ["datasets/text datasets/Buzzfeed Political News Dataset", "en", "politics", "wall-street-journal,the-economist,bbc,NPR,ABC,CBS,USA-today,the-guardian,NBC,washington-post,ending-the-fed,true-pundit,abcnews,dc-gazette,libertywritersnews,before-its-news,infowars,real-news-right-now,the-onion,huff-post-satire,borowitz-report,the-beaverton,SatireWire,faking-news"],
#     "CHECKED-COVID19": ["datasets/text datasets/CHECKED-COVID19.csv", "zh", "health", "white-paper-on-the-social-value-of-chinese-online-medium,research-report-on-the-public-awareness-and-information-dissemination-of-covid-19,weibo-rumor-debunking-system"],
#     "Covid-19-Fake-News-Detection": ["datasets/text datasets/COVID-19 Fake News Dataset/COVID-19 Fake News Dataset/fake_new_dataset.xlsx","en","health", "Webhose.io"],
#     "Covid19-FN": ["datasets/text datasets/COVID19FN/COVID19FN/COVID19FN.csv", "en", "health", "N\A"],
#     "COVID19-Rumor-Data": ["datasets/text datasets/COVID19-Rumor-Data/en_dup.csv", "en", "health", "google-crawler"],
#     "CT-FAN-22-eng-test": ["datasets/text datasets/CT-FAN-22/FakeNews_Task3_2022/FakeNews_Task3_2022/Task3_Test/English_data_test_release_with_rating.csv", "en", "politics,war,health", "N\A"],
#     "CT-FAN-22-ger-test": ["datasets/text datasets/CT-FAN-22/FakeNews_Task3_2022/FakeNews_Task3_2022/Task3_Test/German_data_test_release_with_rating.csv", "de", "politics,war,health", "N\A"],
#     "CT-FAN-22-eng-dev": ["datasets/text datasets/CT-FAN-22/FakeNews_Task3_2022/FakeNews_Task3_2022/Task3_train_dev/Task3_english_dev.csv", "en", "politics,war,health", "N\A"],
#     "CT-FAN-22-eng-train": ["datasets/text datasets/CT-FAN-22/FakeNews_Task3_2022/FakeNews_Task3_2022/Task3_train_dev/Task3_english_training.csv", "en", "politics,war,health", "N\A"],
#     "English-News": ["datasets/text datasets/English-News/fake_or_real_news.csv","en", "politics,health,war", "N\A"],
#     "Fake.br-Corpus(FNC0, FNC1, FNC2)": ["datasets/text+image/text+image/fake.br.corpus.csv", "pt", "economy,science,technology,society,dailynews,politics,religion,TV,celebrities", "Diario-do-Brasil,A-Folha-do-Brasil,The-Jornal-Brasil,Top-Five-TV"],
#     "FakeNewsCorpus-Spanish-train": ["datasets/text datasets/FakeNewsCorpus-Spanish/FakeNewsCorpusSpanish-master/train.xlsx", "es", "science,sport,politics,society,health,environment,international", "ABC,Animal-Político,Aristegui-Noticias,BBC-News,CNN-Spanish,El-Clarín,El-Espectador,El-Financiero,El-Mundo,El-País,El-Universal,Excelsior,Forbes,Huffpost,La-Jornada,La-Vanguardia,Marca,Milenio,MVS-Noticias,Proceso,Tiempo,VerificadoMX,Maldito-Bulo,Caza-Hoax"],
#     "FakeNewsCorpus-Spanish-test": ["datasets/text datasets/FakeNewsCorpus-Spanish/FakeNewsCorpusSpanish-master/test.xlsx", "es", "science,sport,politics,society,health,environment,international", "ABC,Animal-Político,Aristegui-Noticias,BBC-News,CNN-Spanish,El-Clarín,El-Espectador,El-Financiero,El-Mundo,El-País,El-Universal,Excelsior,Forbes,Huffpost,La-Jornada,La-Vanguardia,Marca,Milenio,MVS-Noticias,Proceso,Tiempo,VerificadoMX,Maldito-Bulo,Caza-Hoax"],
#     "FakeNewsCorpus-Spanish-dev": ["datasets/text datasets/FakeNewsCorpus-Spanish/FakeNewsCorpusSpanish-master/development.xlsx", "es", "science,sport,politics,society,health,environment,international", "ABC,Animal-Político,Aristegui-Noticias,BBC-News,CNN-Spanish,El-Clarín,El-Espectador,El-Financiero,El-Mundo,El-País,El-Universal,Excelsior,Forbes,Huffpost,La-Jornada,La-Vanguardia,Marca,Milenio,MVS-Noticias,Proceso,Tiempo,VerificadoMX,Maldito-Bulo,Caza-Hoax"],
#     "FakeNewsNet": ["datasets/text+image/text+image/Fakenewsnet/FakeNewsNet_Dataset", "en", "politics,health,economy,socialissues,entertainment", "Politifact,Gossipcop"],
#     "Fake-News-Urdu": ["datasets/text datasets/Fake-News-Urdu/news.xlsx", "ur", "others", "N\A"],
#     "FA-KES": ["datasets/text datasets/FA-KES-Dataset.csv", "en", "war", "Reuters,Etilaf,SANA,Al-Arabiya,Al-Manar,Al-Ahram,Al-Alam,Al-Araby,Al-Sharq,A-Awsat,Daily-Sabah,TRT,Jordan-Times,The-Lebanese-National-News-Agency,Sputnik,TASS"],
#     "Fang": ["datasets/text datasets/fang/", "en", "politics", "Snopes,Politifact"],
#     "FANG-Covid": ["datasets/text datasets/fang-covid-main/articles", "de", "health", "SueddeutscheZeitung,Tagesspiegel,ZEIT,AnonymousNews,Compact-Online,Contra-Magazin,FreieWelt,Journalistenwatch,Kopp-Report,Politikstube,Pravda-TV,RT-DE,RubikonNews"],
#     "FineFake": ["datasets/text datasets/FineFake.pkl", "en", "politics,entertainment,business,health,society", "Snopes,APNews,CNN,NewYorkTimes,TheWashingtonPost,theCDC"],
#     "FNID-FakeNewsNet-dev": ["datasets/text datasets/FNID-dataset/dataset/fake news detection(FakeNewsNet)/fnn_dev.csv", "en", "politics", "politifact"],
#     "FNID-FakeNewsNet-train": ["datasets/text datasets/FNID-dataset/dataset/fake news detection(FakeNewsNet)/fnn_train.csv", "en", "politics", "politifact"],
#     "FNID-FakeNewsNet-test": ["datasets/text datasets/FNID-dataset/dataset/fake news detection(FakeNewsNet)/fnn_test.csv", "en", "politics", "politifact"],
#     "FNID-LIAR-dev": ["datasets/text datasets/FNID-dataset/dataset/fake news detection(LIAR)/liar_dev.csv", "en", "politics", "politifact"],
#     "FNID-LIAR-train": ["datasets/text datasets/FNID-dataset/dataset/fake news detection(LIAR)/liar_train.csv", "en", "politics", "politifact"],
#     "FNID-LIAR-test": ["datasets/text datasets/FNID-dataset/dataset/fake news detection(LIAR)/liar_test.csv", "en", "politics", "politifact"],
#     "ISOT": ["datasets/text datasets/ISOT", "en", "politics", "politifact"],
#     "Jruvika-Fake-News-Detection": ["datasets/text datasets/Jruvika-Fake-News-Detection/data.csv", "en", "others", "reuters,bbc,NYtimes"],
#     "Kaggle-Fake-and-Real-News": ["datasets/text datasets/Kaggle-Fake-and-Real-News/", "en", "politics", "N\A"],
#     "Kaggle-Fake-News-test": ["datasets/text datasets/Kaggle-Fake-News/test.csv", "en", "others", "N\A"],
#     "Kaggle-Fake-News-train": ["datasets/text datasets/Kaggle-Fake-News/train.csv/train.csv", "en", "others", "N\A"],
#     "Kaggle-Fake-News-Sample": ["datasets/text datasets/Kaggle-Fake-News-Sample/resized_v2.csv", "en", "health,politics,religion", "N\A"],
#     "Kaggle-Fake-or-Real-News": ["datasets/text datasets/Kaggle-Fake-or-Real-News/fake_or_real_news.csv", "en", "crime,politics,war", "N\A"],
#     "Kaggle-Getting-Real-about-Fake-news": ["datasets/text datasets/Kaggle-Getting-Real-about-Fake-news/fake.csv", "en", "politics,technology,health", "N\A"],
#     "LIAR-test": ["datasets/text datasets/LIAR/test.tsv", "en", "politics", "politifact,Channel4"], 
#     "LIAR-train": ["datasets/text datasets/LIAR/train.tsv", "en", "politics", "politifact,Channel4"], 
#     "LIAR-valid": ["datasets/text datasets/LIAR/valid.tsv", "en", "politics", "politifact,Channel4"],
#     "LIAR-PLUS-train" : ["datasets/text datasets/LIAR-PLUS/train2.tsv", "en", "politics", "politifact,Channel4"],
#     "LIAR-PLUS-test" : ["datasets/text datasets/LIAR-PLUS/test2.tsv", "en", "politics", "politifact,Channel4"],
#     "LIAR-PLUS-val" : ["datasets/text datasets/LIAR-PLUS/val2.tsv", "en", "politics", "politifact,Channel4"],
#     "MacIntire": ["datasets/text datasets/MacIntire.csv", "en", "politics", "Kdnugget"],
#     "Misinformation-and-fakenews-and-propaganda": ["datasets/text datasets/Misinformation-and-fakenews-and-propaganda/", "en", "others", "honestarticles,propogandaarticles"],
#     "NewPolitifact": ["datasets/text datasets/newpolitifactdataset.csv", "en", "politics", "politifact"],
#     "Politifact-political-rumor": ["datasets/text datasets/Politifact-political-rumor.tsv", "en", "politics", "politifact"],
#     "Random-Political-News": ["datasets/text datasets/Random-Political-News/Public Data/Random Poltical News Dataset", "en", "politics", "wall-street-journal,the-economist,bbc,NPR,ABC,CBS,USA-today,the-guardian,NBC,washington-post,ending-the-fed,true-pundit,abcnews,dc-gazette,libertywritersnews,before-its-news,infowars,real-news-right-now,the-onion,huff-post-satire,borowitz-report,the-beaverton,SatireWire,faking-news"],
#     "ReCOVery": ["datasets/text+image/text+image/ReCOVery/ReCOVery-master/dataset/recovery-news-data.csv", "en", "health", "NewsGuard,MediaBias"],
#     "RUN-NewsReliability-train": ["datasets/text datasets/RUN-NewsReliability/training_set.json", "es", "politics", "N\A"],
#     "RUN-NewsReliability-test": ["datasets/text datasets/RUN-NewsReliability/test_set.json", "es", "politics", "N\A"],
#     "Snopes-Claims": ["datasets/text datasets/Snopes-claims/Snopes", "en", "business,crime,technology,health,politics", "sixindependentfact-checking,snopes"],
#     "Snopes-rumors": ["datasets/text datasets/Snopes-rumors.tsv", "en", "politics,health,science", "snopes"],
#     "Spanish-Political-News": ["datasets/text datasets/spanish-political-fake-news.csv", "es", "politics", "Público,LaMarea,ElComún"],
#     "True-and-Fake-News-election": ["datasets/text datasets/True-and-Fake-News/DTN_data/2016_election", "en", "politics,government,internationalrelation,economics,law,socialissues", " InfoWars,Yournewswire,BeforeItsNews"],
#     "True-and-Fake-News-brexit": ["datasets/text datasets/True-and-Fake-News/DTN_data/brexit", "en", "politics,government,internationalrelation,economics,law,socialissues", " InfoWars,Yournewswire,BeforeItsNews"],
#     "Truth-seeker-2024": ["datasets/text datasets/TruthSeeker2024/TruthSeeker2023/Truth_Seeker_Model_Dataset.csv", "en", "politics", "politifact"],
#     "UFN-Urdu-Fake-News": ["datasets/text datasets/UFN-Urdu-Fake-News/UFN-Urdu-Fake-News/Urdu-Fake-News-master/UFN/UFN", "ur", "politics", "N\A"],
#     "WELFake": ["datasets/text datasets/WELFake/WELFake_Dataset.csv", "en", "politics", "N\A"],
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

    df["language"] = DATA_PATHS[dataset_name][1]
    df["domain"] = DATA_PATHS[dataset_name][2]
    df["platform"] = DATA_PATHS[dataset_name][3]


    return df[["title", "body", "label", "language", "domain", "platform"]]


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
        futures = {
            executor.submit(process_dataset, name, DATA_PATHS[name][0]): name
            for name in DATA_PATHS
        }
        
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
    final_df.to_json(
        "processed_data/keepup-multilingual.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )
    print(f"Merged dataset saved with {len(final_df)} records and columns: {list(final_df.columns)}")

if __name__ == '__main__':
    main()