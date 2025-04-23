import os
import json
import pandas as pd
import xml.etree.ElementTree as ET

def get_AFND_dataframe(path):
    # Paths
    dataset_path =  os.path.join(path, 'Dataset')
    sources_file = os.path.join(path, 'sources.json')

    # Load sources credibility info
    with open(sources_file, 'r') as f:
        sources_info = json.load(f)

    # List to collect all articles
    articles_data = []

    # Traverse each source folder in the Dataset directory
    for source_folder in os.listdir(dataset_path):
        source_path = os.path.join(dataset_path, source_folder)

        if not os.path.isdir(source_path):
            continue

        # Get credibility from sources.json
        credibility = sources_info.get(source_folder, 'unknown')

        # Path to the scrapped_articles.json
        articles_file = os.path.join(source_path, 'scraped_articles.json')

        if os.path.exists(articles_file):
            with open(articles_file, 'r') as f:
                try:
                    data = json.load(f)
                    articles = data.get('articles', [])
                except json.JSONDecodeError:
                    print(f"Could not decode JSON for {articles_file}")
                    continue

                for article in articles:
                    articles_data.append({
                        'source': source_folder,
                        'credibility': credibility,
                        'title': article.get('title', ''),
                        'text': article.get('text', ''),
                        'published_date': article.get('published date', '')
                    })

    # Convert to DataFrame
    df = pd.DataFrame(articles_data)

    return df

def get_Arabic_Satirical_News(path):

    news_texts = []
    labels = []

    # Iterate through all text files in the folder
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                news_texts.append(content)
                labels.append('fake')

    # Create a DataFrame
    df = pd.DataFrame({
        'body': news_texts,
        'label': labels
    })

    return df

def get_BET_Bend_the_Truth(path):

    data = []

    # Walk through train and test folders
    for split in ['Train', 'Test']:
        for label in ['Fake', 'Real']:
            folder_path = os.path.join(path, split, label)
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read().strip()
                        data.append({
                            'text': content,
                            'label': label.lower(),  # 'fake' or 'real'
                        })

    df_data = pd.DataFrame(data)

    return df_data

def get_BuzzFeed_2017(path):

    articles_folder = os.path.join(path, 'articles')
    csv_path = os.path.join(path, 'overview.csv')   
    
    df_csv = pd.read_csv(csv_path)

    data = []

    for _, row in df_csv.iterrows():
        xml_filename = row['XML']
        veracity = row['veracity']
        xml_path = os.path.join(articles_folder, xml_filename)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            title = root.findtext('title', default='').strip()
            maintext = root.findtext('mainText', default='').strip()

            data.append({
                'title': title,
                'maintext': maintext,
                'veracity': veracity
            })

        except Exception as e:
            print(f"Error parsing {xml_filename}: {e}")

    df_data = pd.DataFrame(data)

    return df_data

def get_BuzzFeed_Political_News(path):

    data = []

    for label in ['Fake', 'Real']:
        body_folder = os.path.join(path, label)
        title_folder = os.path.join(path, f"{label}_titles")

        for filename in os.listdir(body_folder):
            if filename.endswith('.txt'):
                file_number = filename.split('_')[0].replace('.txt', '')
                body_path = os.path.join(body_folder, filename)
                title_path = os.path.join(title_folder, f"{file_number}_{label}.txt")

                try:
                    with open(body_path, 'r', encoding='windows-1252') as body_file:
                        body = body_file.read().strip()
                except Exception as e:
                    body = f"An error occurred reading body: {e}"

                try:
                    with open(title_path, 'r', encoding='windows-1252') as title_file:
                        title = title_file.read().strip()
                except Exception as e:
                    title = f"An error occurred reading title: {e}"

                data.append({
                    'title': title,
                    'body': body,
                    'label': label
                })

    df_data = pd.DataFrame(data)

    return df_data

def get_excel_datasets(path):

    df = pd.read_excel(path)

    return df

def get_FakeNewsNet(path):

    # This will hold our extracted data
    records = []

    # Loop over each label folder
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue

        # Label is the part after the last underscore
        label = folder_name.split('_')[-1]

        # Traverse each subfolder containing the news_article.json
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            json_file_path = os.path.join(subfolder_path, 'news_article.json')

            # Check if the file exists
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Append only if title and text exist
                    if 'title' in data and 'text' in data:
                        records.append({
                            'title': data['title'],
                            'text': data['text'],
                            'label': label
                        })
                except Exception as e:
                    print(f"Error reading {json_file_path}: {e}")

    # Create DataFrame
    df = pd.DataFrame(records)

    return df

def get_FA_KES(path):

    df = pd.read_csv(path , encoding='unicode_escape')

    return df

def get_fang(path):

    path_real = os.path.join(path, 'fang_real.csv') 
    path_fake = os.path.join(path, 'fang_fake.csv') 

    fang_fake = pd.read_csv(path_fake)
    fang_real = pd.read_csv(path_real)

    fang_real['label'] = 'real'
    fang_fake['label'] = 'fake'

    # Concatenate the two datasets
    df = pd.concat([fang_real, fang_fake]).reset_index()

    return df

def get_FANG_Covid(path):

    # List to store all the data
    data = []

    # Loop through all files in the folder
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            file_path = os.path.join(path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    content = json.load(file)
                    header = content.get("header", "")
                    article = content.get("article", "")
                    label = content.get("label", "")
                    data.append({"header": header, "article": article, "label": label})
                except json.JSONDecodeError as e:
                    print(f"Error reading {filename}: {e}")

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    return df

def get_FineFake(path):

    fine_fake = pd.read_pickle(path)

    return fine_fake

def get_ISOT(path):

    path_real = os.path.join(path, 'True.csv') 
    path_fake = os.path.join(path, 'Fake.csv') 

    isot_fake = pd.read_csv(path_fake)
    isot_real = pd.read_csv(path_real)

    isot_real['label'] = 'True'
    isot_fake['label'] = 'Fake'

    # Concatenate the two datasets
    df = pd.concat([isot_real, isot_fake]).reset_index()

    return df

def get_Kaggle_Fake_and_Real_News(path):

    #path_real = os.path.join(path, 'True.csv') 
    path_fake = os.path.join(path, 'Fake.csv') 

    fake = pd.read_csv(path_fake)
    #real = pd.read_csv(path_real)

    #real['label'] = 'True'
    fake['label'] = 'Fake'

    # Concatenate the two datasets
    #df = pd.concat([real, fake]).reset_index()

    return fake

def get_Kaggle_Fake_News_test(path):

    test_path = os.path.join(path, 'test.csv') 
    labels_path = os.path.join(path, 'submit.csv')

    test = pd.read_csv(test_path, index_col='id')
    labels = pd.read_csv(labels_path)

    kfn = test.merge(labels, on='id')

    return kfn

def get_LIAR(path):

    column_names = ["file_name", "label", "title", '1','2','3','4','5','6','7','8','9','10','11']
    liar = pd.read_csv(path, sep='\t', header=None, names=column_names)

    return liar

def get_LIAR_PLUS(path):

    column_names = ["file_name", "label", "title", '1','2','3','4','5','6','7','8','9','10','11','body']
    liar = pd.read_csv(path, sep='\t', header=None, names=column_names)

    return liar

def get_Misinformation_and_fakenews_and_propaganda(path):

    path_real = os.path.join(path, 'DataSet_Misinfo_FAKE.csv','DataSet_Misinfo_FAKE.csv') 
    path_fake = os.path.join(path, 'DataSet_Misinfo_TRUE.csv','DataSet_Misinfo_TRUE.csv') 

    fake = pd.read_csv(path_fake)
    real = pd.read_csv(path_real)

    real['label'] = 'real'
    fake['label'] = 'fake'

    # Concatenate the two datasets
    df = pd.concat([real, fake]).reset_index()

    return df

def get_Politifact_political_rumor(path):

    column_names = ["Claim Text", "Veracity", "Category", "Published Date"]
    ppr = pd.read_csv(path, sep='\t', names=column_names)

    return ppr

def get_RUN_NewsReliability(path):

    with open(path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Initialize a list to store records
    records = []

    # Iterate through each key (like "0", "1", etc.)
    for key, value in raw_data.items():

        text = value.get("TEXT", "").strip()
        title = value.get("TITLE", "").strip()
        alba = value.get("VALUE_ALBA", "")

        
        # Combine all into a single dict (flattened)
        record = {
            "title": title,
            "text": text,
            "label": alba
        }
        
        records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(records)

    return df

def get_Snopes_Claims(path):

    extracted_data = []

    # Loop through all files in the folder
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            file_path = os.path.join(path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    claim = data.get("Claim", "")
                    description = data.get("Description", "")
                    credibility = data.get("Credibility", "")
                    record = {
                        'title' : claim, 
                        'body' : description, 
                        'label' : credibility
                    }
                    extracted_data.append(record)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(extracted_data)

    return df

def get_Spanish_Political_News(path):

    spn = pd.read_csv(path, sep=';')

    return spn

def get_True_and_Fake_News(path):

    column_names = ["label", "text"]
    combined_df = pd.DataFrame(columns=column_names)

    for fname in os.listdir(path):
        if fname.endswith(".txt"):
            file_path = os.path.join(path, fname)

            # Use Python engine to handle irregular separators
            df = pd.read_csv(file_path, sep="#", header=None, names=column_names, engine='python',  on_bad_lines='skip')

            # Replace tab characters with spaces
            df = df.map(lambda x: x.replace('\t', ' ') if isinstance(x, str) else x)

            combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

def get_UFN_Urdu_Fake_News(path):

    data = []

    # Loop through folders labeled '0' and '1'
    for label in ['0', '1']:
        folder_path = os.path.join(path, label)
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Read only text files
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().replace('\t', ' ')  # Optional: remove tabs
                    data.append({'text': text, 'label': int(label)})

    # Create DataFrame
    df = pd.DataFrame(data)

    return df