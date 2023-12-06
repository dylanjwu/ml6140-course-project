import sys
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from mlxtend.evaluate import bias_variance_decomp
from imblearn.over_sampling import SMOTE
from preprocess import write_pca_dataframe, write_all_dataframes, get_df_from_db

df_snippets = get_df_from_db() # get all data from database


def run_model(model, X, y, samples_map, filename):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predicted = cross_val_predict(model, X, y, cv=kf)
    print(f'      predicted: {predicted}')
    scores = cross_val_score(model, X, y, cv=kf)
    mean_score = scores.mean()
    print("      Cross-Validation Scores:", scores)
    print("      Mean Score:", scores.mean())
    print("      Standard Deviation of Scores:", scores.std())

    bias = np.mean(y != predicted)  
    variance = np.var(predicted != y) 
    print(f"      Variance of Misclassifications: {variance}")

    # Get misclassified indexes
    misclassified_indexes = np.where(y != predicted)[0]
    misclassified_samples = samples_map.iloc[misclassified_indexes] 
    misclassified_samples.loc[:, 'Predicted'] = predicted[misclassified_indexes]

    samples_filename = f'.{"".join(filename.split(".")[:-1])}_misclassified.csv'
    print(f'samples_filename: {samples_filename}')
    misclassified_samples.to_csv(samples_filename)

    return mean_score



def get_data(filename="tfidf.csv", labels_name='___LANGUAGE___', classes=[]):
    df = pd.read_csv(filename)

    print(f'shape of data: {df.shape}')

    if len(classes) > 0:
        df = df[df['___LANGUAGE___'].isin(classes)]

    columns_to_exclude = {')', '>', ']', '}'} # remove vars causing multicollinearity
    columns = [col for col in df.columns if col not in columns_to_exclude]
    df = df[columns]

    labels = df[labels_name]
    df = df.drop(labels_name, axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42) 
    X_train, y_train = df, labels # all the data (for cross validation)

    X_train = X_train.drop(['Unnamed: 0'], axis=1)
    df.dropna(axis=1, inplace=True)

    return (X_train, y_train)

def run_all(preprocess_files_first=True):
    """
        Configurations:
            preprocessing, create files:
                {TF-IDF, BoW} x {Regular, special_chars} x {100 features} x {1000 rows}
                For TF-IDF regular --> also do 200 features, 100 features, 50 features
                PCA based on most successful configurations above, do 70, 90, 100 (on 1-2 features)
                TOTAL CONFIGS: 4 models * (4+3+3) + 3 models * (3 to 6) = 40-60 total configs
            (initialize with) Gridsearch for MLP, SVM, DT
            BoW and TF-IDF:
                Regular and special_chars:
                    (for regular TF-IDF: 200 features, 100 features, 50 features)
                    Models: NB, SVM, DT, MLP
            PCA (70, 90, 100) (based on most successful)
                Models: SVM, DT, MLP
            (based on results, redo grid search if necessary on MLP, SVM, DT)
            
    """


    models = {
        'Naive Bayes': MultinomialNB(),
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', 
                    max_depth=20, 
                    min_samples_leaf=4, 
                    min_samples_split=2, 
                    splitter='best'),
        'SVM': SVC(C=10, gamma=1, kernel='linear'),
        'MLP': MLPClassifier(hidden_layer_sizes=(96, 100), max_iter=1000, random_state=42, activation='logistic'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    models_for_pca = {key: value for key,value in models.items() if key != 'Naive Bayes'} # don't use NB for PCA

    vectorizers = ['bow', 'tfidf']
    token_types = ['', 'special_chars']
    rows = [1000]
    features = [50, 100]
    pca_components = [70, 80, 100]

    print(f'preprocess: {preprocess_files_first}')
    if preprocess_files_first: # write files to data dir first
        for r in rows:
            for f in features:
                write_all_dataframes(df_snippets, r, f)

    for model_name, model in models.items():
        print(f'MODEL: {model_name}')
        for v in vectorizers:
            for tt in token_types:
                for row_c in rows:
                    for ftr_c in features:
                        if tt == '':
                            filename = f'./data/{v}_{ftr_c}features_{row_c}rows.csv'
                        else:
                            filename = f'./data/{v}_{tt}_{30}features_{row_c}rows.csv'
                        X, y = get_data(filename=filename)
                        print(f'   [{v} - {tt} - rows:{row_c} - features:{ftr_c} ]')
                        samples_map = pd.read_csv(f'./data/cleaned_samples_{row_c}rows.csv')
                        run_model(model, X, y, samples_map, filename)
                    
                    if model_name != 'Naive Bayes':
                        if tt == '':
                            for pc in  [70, 80, 100]:
                                pca_filename = write_pca_dataframe(vectorizer=v, token_type=tt,  N=row_c, components=pc)
                                X, y = get_data(filename=pca_filename)
                                print(f'   [PCA - {v} - {tt} - rows:{row_c} - components:{pc} ]')
                                samples_map = pd.read_csv(f'./data/cleaned_samples_{row_c}rows.csv')
                                run_model(model, X, y, samples_map, pca_filename)
                        else:
                            for pc in  [10, 20, 30]:
                                pca_filename = write_pca_dataframe(vectorizer=v, token_type=tt,  N=row_c, components=pc)
                                X, y = get_data(filename=pca_filename)
                                print(f'   [PCA - {v} - {tt} - rows:{row_c} - components:{pc} ]')
                                samples_map = pd.read_csv(f'./data/cleaned_samples_{row_c}rows.csv')
                                run_model(model, X, y, samples_map, pca_filename)




if __name__ == '__main__':

    # arguments = sys.argv[1:]

    # if len(arguments) < 1:
    #     print('No filename argument provided; exiting')

    # filename = arguments[0]


    # with open('./output/all_output.txt', 'w') as f:
    #     sys.stdout = f
    run_all(preprocess_files_first=False)