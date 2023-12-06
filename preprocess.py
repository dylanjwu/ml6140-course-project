"""
    Convert the raw data into model-friendly format (including ITF-DF)
    soures: 
        https://towardsdatascience.com/classification-model-for-source-code-programming-languages-40d1ab7243c2
"""
import sys
import pandas as pd
import sqlite3
import numpy as np
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

""" START OF DATA CLEANING FUNCTIONS """

def get_special_chars_df(snippets):
    chars = set('(){}*&%$#@!-+=[]<>?"\'/\\,|:;^~`')

    processed_snippets = snippets.apply(lambda snippet: ''.join(ch for ch in snippet if ch in chars))
    
    return processed_snippets

def remove_comments(snippet):
    try:
        txt = re.sub(r'//.*?\n', '', snippet)
        txt = re.sub(r'/\*.*?\*/', '', txt, flags=re.DOTALL)
        txt = re.sub(r'//.*?\n', '', txt)
        txt = re.sub(r'/\*.*?\*/', '', txt, flags=re.DOTALL)
        txt = re.sub(r'#.*?\n', '', txt)
        txt = re.sub(r"'''(.*?)'''", '', txt, flags=re.DOTALL)
        txt = re.sub(r'"""(.*?)"""', '', txt, flags=re.DOTALL)
    except Exception as e: # should not happen
        print("Failed snippet: ")
        print(snippet)
        print(e)
        return None

    return txt

def remove_stop_words(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def clean(df):
    df = df.where(pd.notnull(df), None).dropna().replace('', np.nan).dropna() #remove null/nan/empty values

    for i,row in df.iterrows():
        # if i%10000 == 0:
        #     print(i)
        snippet = df.at[i, 'snippet']
        no_comments_snippet = remove_comments(snippet)
        no_comments_snippet = remove_stop_words(no_comments_snippet)
        # no_new_l = re.sub(r'\n', '', no_comments_snippet) # remove newlines
        cleaned_snippet = re.sub(r'\b([A-Za-z])\1+\b', '', no_comments_snippet) # same character vars, single character -- useless, remove
        cleaned_snippet = re.sub(r'\b[A-Za-z]\b', '', cleaned_snippet)

        df.at[i, 'snippet'] = cleaned_snippet
    
    # return df.dropna(inplace=True)
    return df


""" END OF DATA CLEANING FUNCTIONS """

""" START OF VECTORIZATION FUNCTIONS """

def tfidf_vectorize(df, max_features=100, rows=10000):
    print('tfidf_vectorize function')

    pattern = r"""([A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`"'])"""
    vectorizer = TfidfVectorizer(token_pattern=pattern, max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(df['snippet'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    tfidf_df['___LANGUAGE___'] = df['language']

    tfidf_df.to_csv(f"./data/tfidf_{max_features}features_{rows}rows.csv", index=True)
    print("Task completed")

def bag_of_words_vectorize(df, max_features=100, rows=10000):
    print('bag_of_words_vectorize function')

    pattern = r"""([A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`"'])"""
    vectorizer = CountVectorizer(token_pattern=pattern, max_features=max_features) # gets the most frequent tokens (limited to 100)
    bow_matrix = vectorizer.fit_transform(df['snippet'])
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    bow_df['___LANGUAGE___'] = df['language']

    bow_df.to_csv(f"./data/bow_{max_features}features_{rows}rows.csv", index=True)
    print("Task completed")

def bag_of_words_chars_vectorize(df, max_features=100, rows=10000):
    print('bag_of_words_chars_vectorize function')

    vectorizer = CountVectorizer(analyzer='char', max_features=max_features)    
    just_special_chars = get_special_chars_df(df['snippet'])
    bow_matrix = vectorizer.fit_transform(just_special_chars)
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    bow_df['___LANGUAGE___'] = df['language']

    bow_df.to_csv(f"./data/bow_special_chars_{30}features_{rows}rows.csv", index=True)
    print("Task completed")

def tfidf_chars_vectorize(df, max_features=100, rows=10000):
    print('tfidf_chars_vectorize function')

    vectorizer = TfidfVectorizer(analyzer='char', max_features=max_features)
    just_special_chars = get_special_chars_df(df['snippet'])

    tfidf_matrix = vectorizer.fit_transform(just_special_chars)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    

    tfidf_df['___LANGUAGE___'] = df['language']

    tfidf_df.to_csv(f"./data/tfidf_special_chars_{30}features_{rows}rows.csv", index=True)
    print("Task completed")

""" END OF VECTORIZATION FUNCTIONS """

def get_df_from_db():
    path = '/Users/dylanwu/Downloads/snippets-dev/snippets-dev.db'
    conn = sqlite3.connect(path)

    query = 'SELECT * FROM snippets_subset'

    snippets = pd.read_sql_query(query, conn)

    conn.close()
    return snippets

def write_all_dataframes(df_snippets, N=1000, features=100):

    limited_df = df_snippets.groupby('language').apply(lambda x: x.sample(min(len(x), N)))
    limited_df = limited_df.reset_index(drop=True)
    limited_df = clean(limited_df)
    limited_df.to_csv(f'./data/cleaned_samples_{N}rows.csv')

    vectorizers = [tfidf_vectorize, tfidf_chars_vectorize, bag_of_words_vectorize, bag_of_words_chars_vectorize]

    for vectorizer in vectorizers:
        vectorizer(limited_df, max_features=features, rows=N)

def write_pca_dataframe(vectorizer='tfidf', token_type='',  N=1000, components=100):
    if token_type == '':
        original_filename = f'./data/{vectorizer}_{100}features_{N}rows.csv'
    else:
        original_filename = f'./data/{vectorizer}_{token_type}_{30}features_{N}rows.csv'

    vectorized_df = pd.read_csv(original_filename)

    if '___LANGUAGE___' in vectorized_df.columns:
        language_df = vectorized_df['___LANGUAGE___']
        vectorized_df = vectorized_df.drop('___LANGUAGE___', axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(vectorized_df)

    print(f'X_scaled shape: {X_scaled.shape}')

    pca = PCA(n_components=components)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca)
    pca_df['___LANGUAGE___'] = language_df

    if token_type == '':
        filename = f'./data/{vectorizer}_pca_{components}features_{N}rows.csv'
    else:
        filename = f'./data/{vectorizer}_{token_type}_pca_{components}features_{N}rows.csv'

    pca_df.to_csv(filename)

    return filename

if __name__ == '__main__':
    
    write_all_dataframes()

