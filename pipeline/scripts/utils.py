import numpy as np
import pandas as pd
# from sklearn import metrics
# import seaborn as sn
# import matplotlib.pyplot as plt
import textwrap
from typing import List

from sklearn.feature_extraction.text import  TfidfVectorizer # type: ignore
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import nltk
from typing import Protocol
from nltk.stem import WordNetLemmatizer # type: ignore
from nltk.corpus import stopwords # type: ignore

# scientific units
from quantities import units # type: ignore
# from wordcloud import WordCloud

def preprocess_text(data: pd.DataFrame, 
                    remove_punctuation: bool = False, 
                    lower_case: bool = True, 
                    remove_digits: bool = False) -> pd.DataFrame:
    '''preprocesses the text data (in place - no return value here)

    Arguments:
        data = DataFrame with the data
        remove_punctuation = remove punctuation (default = False)
        lower_case = convert to lower case (default = True)
        remove_digits = remove digits (default = False)
    '''
    if remove_punctuation:
        punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~'   # `|` is not present here
        transtab = str.maketrans(dict.fromkeys(punct, ' '))
        data['title_clean'] = data['title'].str.translate(transtab)
        data['abstract_clean'] = data['abstract'].str.translate(transtab)
    if lower_case:
        if 'title_clean' in data.columns:
            data['title_clean'] = data['title_clean'].str.lower()
            data['abstract_clean'] = data['abstract_clean'].str.lower()
        else:
            data['title_clean'] = data['title'].str.lower()
            data['abstract_clean'] = data['abstract'].str.lower()
    if remove_digits:
        if 'title_clean' in data.columns:
            data['title_clean'] = data['title_clean'].str.replace(r'\d+', '', regex=True)
            data['abstract_clean'] = data['abstract_clean'].str.replace(r'\d+', '', regex=True)
        else:
            data['title_clean'] = data['title'].str.replace(r'\d+', '', regex=True)
            data['abstract_clean'] = data['abstract'].str.replace(r'\d+', '', regex=True)
    # collapse spaces
    data['title_clean'] = data['title_clean'].str.replace(r'\s+', ' ', regex=True)
    data['abstract_clean'] = data['abstract_clean'].str.replace(r'\s+', ' ', regex=True)


def print_paper(pmid: int, 
                df: pd.DataFrame, 
                width: int = 80) -> None:
    '''Prints a paper in readable format.

    Arguments:
        pmid = The pubmed-id of the paper
        df = DataFrame with papers
        width = width of the text
    '''
    # check wheter a paper with this pmid exists
    if pmid not in df['pmid'].values:
        print(f'Paper with pmid {pmid} not found.')
        return
    record = df[df['pmid']==pmid]
    # get the index of the record
    record_index = record.index[0]
    print(f'Paper {record_index}:\n')
    print(record)
    print(f'PubMed ID: {pmid}')
    print(f'\nTitle: ')
    print("\n".join(textwrap.wrap(record.loc[record_index, 'title'], width = width)))
    print(f'\nAbstract:')
    print("\n".join(textwrap.wrap(record.loc[record_index, 'abstract'], width = width)))

## find terms in either title or abstract column of baseline_train_test
def find_pesticide_terms(df:pd.DataFrame, 
                         terms:List[str]) -> pd.DataFrame:
    """
    Find pesticide terms in rows of the DataFrame. Searches for terms in both the title and abstract columns.
    It does so in a case-insensitive manner and returns the rows that contain any of the terms.

    Args:
        df (pd.DataFrame): The DataFrame to search.
        terms (list): A list of terms to search for.
        
    Returns:
        pd.DataFrame: A DataFrame containing rows where the specified column contains any of the terms.
    """
    terms = [term.lower() for term in terms]
    #print(f'Finding terms: {terms}')
    # Check if the columns exist in the DataFrame
    if 'title_clean' not in df.columns or 'abstract_clean' not in df.columns:
        raise ValueError("DataFrame must contain 'title' and 'abstract' columns.")
    # Filter the DataFrame based on the presence of terms in either the title or abstract
    # also lower the title and abstract columns
    matches = df[df['title_clean'].str.contains('|'.join(terms), na=False) | 
              df['abstract_clean'].str.contains('|'.join(terms), na=False)]
    
    # create a new dataframe with the pmid and the title and the found terms
    found_terms = pd.DataFrame()
    # make a selection of the df dataframe based on the pmid column of the matches dataframe
    found_terms = df[df['pmid'].isin(matches['pmid'])].copy()

    # # copy original data from bigger dataframe
    # found_terms['pmid'] = matches['pmid']
    # found_terms['title'] = matches['title']
    # found_terms['abstract'] = matches['abstract']
    # found_terms['title_clean'] = matches['title_clean']
    # found_terms['abstract_clean'] = matches['abstract_clean']

    found_terms['found_terms_title'] = matches['title_clean'].str.findall('|'.join(terms))
    found_terms['found_terms_abstract'] = matches['abstract_clean'].str.findall('|'.join(terms))

    # only keep unique values in the new columns
    found_terms['found_terms_title'] = found_terms['found_terms_title'].apply(lambda x: list(set(x)))
    found_terms['found_terms_abstract'] = found_terms['found_terms_abstract'].apply(lambda x: list(set(x)))
    found_terms['found_terms'] = found_terms['found_terms_title'] + found_terms['found_terms_abstract']

    # delete found_terms_title and found_terms_abstract
    found_terms = found_terms.drop(columns=['found_terms_title', 'found_terms_abstract'])
    return found_terms


class SupportsVectorizer(Protocol):
    def fit(self, raw_documents, y=None): ...
    def transform(self, raw_documents): ...
    def fit_transform(self, raw_documents, y=None): ...

def make_vectorizer(vocabulary_size: int = 2000, type="TF-IDF") -> SupportsVectorizer:
    '''Creates an (TF-IDF) vectorizer. Calls the get_stopwords() function to get the stopwords and 
    uses the custom Tokenizer class of this package for tokenization.

    Arguments:
        vocabulary_size = size of the vocabulary (default = 2500)
    Returns:
        Vectorizer instance (only TfidfVectorizer for now)
    '''
    if not type == "TF-IDF":
        raise ValueError(f"Unknown vectorizer type: {type}. Currently, only TF-IDF is supported.")
    stopwords = get_stopwords()
    tokenizer = Tokenizer(stop_words=stopwords, min_length=3)
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        max_features=vocabulary_size,
    )
    return vectorizer


def make_embeddings(vectorizer: SupportsVectorizer,
                           papers: pd.DataFrame, 
                           column: str = 'abstract_clean',
                           only_transform: bool = False):
    '''Creates (TF-IDF) embeddings for the papers.
    Arguments:
        vectorizer = (TF-IDF) vectorizer, or any object that supports the SupportsVectorizer protocol.
        papers = DataFrame with the papers
        column = column to use for the embeddings (default = 'abstract_clean')
        only_transform = whether to only transform() the data, or call fit_transform() (default = False)
    '''
    if column not in papers.columns:
        raise ValueError(f"Column {column} not found in DataFrame.")
    if only_transform:
        # transform the data
        X = vectorizer.transform(papers[column])
    else:
        # fit and transform the data
        X = vectorizer.fit_transform(papers[column])
    return X


def get_stopwords(extended: bool = True, 
                  add_scientific_units: bool = True, 
                  custom: List[str] = None):
    '''Constructs a set of stopwords, based on nltk English stopwords (stopwords.words('english')) and returns this as a list
    Arguments:
        extended = whether to use extended stopwords (default = True). 
                    The extended set includes 'said', 'would', 'could', 'told', 'also', 'one', 'two', 'three', 'study', 'result', 'method',
                    'used', 'using', 'wa', 'use', 'sup', '/sup', 'sub', '/sub'
        add_scientific_units = whether to add scientific units (default = True)
                    these units come from the quantities package
        custom = list of custom stopwords to be added (default = None)
    '''
    stops = set(stopwords.words('english'))
    if extended:
        stops = stops.union({
            'said', 'would', 'could', 'told', 'also', 'one', 'two', 'three', 'study', 'result', 'method',
            'used', 'using', 'wa', 'use', 'sup', '/sup', 'sub', '/sub'})
    if add_scientific_units:
        unit_symbols = [u.symbol for _, u in units.__dict__.items() if isinstance(u, type(units.deg))]
        stops = stops.union(unit_symbols)
    if custom:
        stops = stops.union(custom)
    stops = list(stops)
    return stops

# class StemTokenizer:
#     def __init__(self):
#         self.porter = PorterStemmer()
#     def __call__(self, doc):
#         tokens = word_tokenize(doc)
#         return [self.porter.stem(t) for t in tokens]
    
class Tokenizer:
    '''class for tokenizing text data using nltk.word_tokenize()'''
    def __init__(self, min_length=2, lemmatize=True, stop_words=get_stopwords(), remove_digits=True):
        self.min_length = min_length
        self.lemmatize = lemmatize
        self.stop_words = stop_words
        self.remove_digits = remove_digits

    def __call__(self, s):
        return self.tokenize_text(s)
    
    def __str__(self):
        return f'Tokenizer(min_length={self.min_length}, lemmatize={self.lemmatize}, # stop_words={len(self.stop_words)}, remove_digits={self.remove_digits})'
    def __repr__(self):
        return f'Tokenizer(min_length={self.min_length}, lemmatize={self.lemmatize}, # stop_words={len(self.stop_words)}, remove_digits={self.remove_digits})'
        
    def tokenize_text(self, s):
        '''tokenizes the text data'''
        # split string into words (tokens)
        tokens = nltk.tokenize.word_tokenize(s)

        # remove short words, they're probably not useful
        tokens = [t for t in tokens if len(t) >= self.min_length]

        if self.lemmatize:
            lemmatizer = WordNetLemmatizer()
            # put words into base form
            tokens = [lemmatizer.lemmatize(t) for t in tokens]

        if self.stop_words:
            # remove stopwords
            tokens = [t for t in tokens if t not in self.stop_words]
        if self.remove_digits:
            # remove any digits, i.e. "3rd edition"
            tokens = [t for t in tokens if not any(c.isdigit() for c in t)]

        return tokens

#new_pesticide_papers = utils.find_nearest_neighbors(pesticide_papers, new_pesticide_papers, pesticide_paper_embeddings, new_pesticide_paper_embeddings)
def find_nearest_neighbors(reference_papers: pd.DataFrame, 
                            query_papers: pd.DataFrame, 
                            reference_papers_embeddings: np.ndarray, 
                            query_papers_embeddings: np.ndarray, 
                            top_n_papers: int = 5) -> pd.DataFrame:
    '''Finds the nearest neighbors of the new papers based on the cosine similarity of the TF-IDF matrix.
    Arguments:
        reference_papers = DataFrame with the reference papers
        query_papers = DataFrame with the query papers
        reference_papers_embeddings = TF-IDF sparse matrix with the reference papers
        query_papers_embeddings = TF-IDF sparse matrix with the query papers
        top_n_papers = number of recommendations to return (default = 5)
    Returns:
        DataFrame with the top_n_papers recommendations for each query paper
    '''
    # generate a mapping from paper pubmed id -> index (in df)
    pmid2df_index = pd.Series(reference_papers.index, index=reference_papers['pmid'])
    print(f"Paper2idx mapping: {pmid2df_index}")
    return query_papers

    # # calculate the pairwise similarities for this movie
    # scores = cosine_similarity(new_paper_embeddings, paper_embeddings)
    
    # # get the indexes of the highest scoring movies
    # # get the first K recommendations
    # # don't return itself!
    # top_scoring_idx = (-scores).argsort(axis=1)[:, 1:top_n_papers+1]
    # top_scoring_scores = scores[np.arange(scores.shape[0])[:, None], top_scoring_idx]

    # best_matches_pmids = papers['pmid'].iloc[top_scoring_idx] # fetches a Series
    # best_matches_titles = papers['title'].iloc[top_scoring_idx]    
    # best_matches = pd.concat([best_matches_pmids, best_matches_titles], axis=1)
    # best_matches['score'] = top_scoring_scores
    # return best_matches


# def plot_top_words(model, feature_names, n_components=10, n_top_words=10):
#     '''plots the top words for each topic'''
#     import math
#     rows = 2
#     cols = math.ceil(n_components / rows)
#     fig, axes = plt.subplots(rows, cols, figsize=(30, 15), sharex=True)
#     axes = axes.flatten()
#     for topic_idx, topic in enumerate(model.components_):
#         top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
#         top_features = [feature_names[i] for i in top_features_ind]
#         weights = topic[top_features_ind]

#         ax = axes[topic_idx]
#         ax.barh(top_features, weights, height=0.7)
#         ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
#         ax.invert_yaxis()
#         ax.tick_params(axis="both", which="major", labelsize=20)
#         for i in "top right left".split():
#             ax.spines[i].set_visible(False)
#             fig.suptitle('Topic Top Words', fontsize=40)

#     plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
#     plt.show()


# def evaluate_model(model, x_train, y_train, x_test, y_test, verbose = True):
#     '''evaluates the already fitted model and prints the results'''
#     # print('======')
#     # print(y_test[1:10])
#     # print('======')
#     # print(y_train[1:10])
#     # print('======')
#     y_pred = model.predict(x_test)

#     results = {}

#     num_digits = 5
#     train_accuracy = model.score(x_train, y_train)
#     results['train_accuracy'] = train_accuracy

#     test_accuracy = metrics.accuracy_score(y_test, y_pred)
#     results['accuracy'] = test_accuracy

#     test_precision = metrics.precision_score(y_test, y_pred)
#     results['precision'] = test_precision

#     test_recall = metrics.recall_score(y_test, y_pred)
#     results['recall'] = test_recall

#     test_f1 = metrics.f1_score(y_test, y_pred)
#     results['f1'] = test_f1

#     cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
#     results['confusion_matrix'] = cnf_matrix

#     if verbose:
#         print("train accuracy:", round(train_accuracy, num_digits))
#         print("test accuracy: ", round(test_accuracy, num_digits))
#         print("test precision:", round(test_precision, num_digits))
#         print("test recall:   ", round(test_recall, num_digits))
#         print("test F1 score: ", round(test_f1, num_digits))
#         print("confusion matrix:")
#         print(cnf_matrix)
#     return results

    
# # Scikit-Learn is transitioning to V1 (and can plot this itself),
# # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
# # but it's not available on Colab yet 
# # The changes modify how confusion matrices are plotted
# def plot_confusion_matrix(cm, class_labels):
#     plt.figure(figsize=(4, 3))
#     #sn.set_theme(rc={'figure.figsize':(7, 5)}) #(11.7,8.27)
#     df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
#     ax = sn.heatmap(df_cm, annot=True, fmt='g')
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("Target")

# def visualize(df, label):
#     '''visualizes the data with a wordcloud.
    
#     Arguments:
#         df = DataFrame with the data
#         label = label to visualize'''
#     words = ''
#     for msg in df[df['text_label'] == label]['abstract_clean']:
#         words += msg + ' '
#     make_wordcloud(words)


# def make_wordcloud(words):
#     '''Takes a string of space-separated words to generatye a wordcloud'''
#     wordcloud = WordCloud(width=600, height=400).generate(words)
#     plt.imshow(wordcloud)
#     plt.axis('off')
#     plt.show()


# def sample_validation_set(positives_file, negatives_file, 
#                           n_positive=30, n_negative=470,
#                           out_file='validation_set.csv',
#                           abstract_required=True):
#     '''Samples a validation set from positive and negative samples'''

#     # read positive and negative samples
#     #print(f'positives={positives_file}, negatives={negatives_file}')

#     positives = pd.read_csv(positives_file, sep='\t', na_values="NA")
#     negatives = pd.read_csv(negatives_file, sep='\t', na_values="NA")

#     if abstract_required:
#         # filter out rows without abstract
#         positives = positives[positives['abstract'].notna()]
#         negatives = negatives[negatives['abstract'].notna()]

#     # sample validation set
#     validation_positives = positives.sample(n=n_positive)
#     validation_negatives = negatives.sample(n=n_negative)

#     # add label
#     validation_positives['label'] = 1
#     validation_negatives['label'] = 0
#     validation_positives['text_label'] = 'pest'
#     validation_negatives['text_label'] = 'contr'

#     #concatenate
#     validation_set = pd.concat([validation_positives, validation_negatives])

#     #remove duplicates (in set pest and set contr)
#     validation_set = validation_set.drop_duplicates()

#     # shuffle
#     validation_set = validation_set.sample(frac=1)

#     # write validation set
#     validation_set.to_csv(out_file, sep='\t', index=False)


#     # create a function that generates recommendations
# def recommend(pmid, df, paper2idx, X, top_n_papers = 5):
#     ''' Recommends similar papers based on the cosine similarity of the TF-IDF matrix.

#     Arguments:
#         pmid: pubmed-id
#         df: DataFrame with papers that was used to build X
#         paper2idx: mapping of pubmedid to dataframe index
#         X: TF-IDF sparse matrix
#         top_n_papers: number of recommendations to return (default = 5)
#     '''
#     # will skip the first (self hit)
#     top_n_papers += 1
#     idx = paper2idx[pmid]
#     if type(idx) == pd.Series: # when there are multiple with the same pmid
#         idx = idx.iloc[0]
    
#     # calculate the pairwise similarities for this movie
#     query = X[idx]
#     scores = cosine_similarity(query, X)
    
#     # currently the array is 1 x N, make it just a 1-D array
#     scores = scores.flatten()
    
#     # get the indexes of the highest scoring movies
#     # get the first K recommendations
#     # don't return itself!
#     top_scoring_idx = (-scores).argsort()[1:top_n_papers]
#     top_scoring_scores = scores[top_scoring_idx]

#     best_matches_pmids = df['pmid'].iloc[top_scoring_idx] # fetches a Series
#     best_matches_titles = df['title'].iloc[top_scoring_idx]    
#     best_matches = pd.concat([best_matches_pmids, best_matches_titles], axis=1)
#     best_matches['score'] = top_scoring_scores
#     return best_matches