import pandas as pd
# from sklearn import metrics
# import seaborn as sn
# import matplotlib.pyplot as plt
import textwrap
from typing import List

# import nltk
# from nltk.stem import WordNetLemmatizer, PorterStemmer
# from nltk.corpus import stopwords
# # scientific units
# from quantities import units
# from wordcloud import WordCloud

# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

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
    return df[df['title'].str.contains('|'.join(terms), na=False) | 
              df['abstract'].str.contains('|'.join(terms), na=False)]


# def get_stopwords(extended=True, add_scientific_units=True, custom=None):
#     '''constructs a set of stopwords and returns this as a list'''
#     stops = set(stopwords.words('english'))
#     if extended:
#         stops = stops.union({
#             'said', 'would', 'could', 'told', 'also', 'one', 'two',
#             'mr', 'new', 'year', 'used'
#         })
#     if add_scientific_units:
#         unit_symbols = [u.symbol for _, u in units.__dict__.items() if isinstance(u, type(units.deg))]
#         ## add scientific units to stopwords
#         stops = stops.union(unit_symbols)
#     if custom:
#         stops = stops.union(custom)
#     stops = list(stops)
#     return stops

# class StemTokenizer:
#     def __init__(self):
#         self.porter = PorterStemmer()
#     def __call__(self, doc):
#         tokens = word_tokenize(doc)
#         return [self.porter.stem(t) for t in tokens]
    
# class Tokenizer:
#     '''class for tokenizing text data using nltk.word_tokenize()'''
#     def __init__(self, min_length=2, lemmatize=True, stop_words=get_stopwords(), remove_digits=True):
#         self.min_length = min_length
#         self.lemmatize = lemmatize
#         self.stop_words = stop_words
#         self.remove_digits = remove_digits

#     def __call__(self, s):
#         return self.tokenize_text(s)
    
#     def tokenize_text(self, s):
#         '''tokenizes the text data'''
#         # split string into words (tokens)
#         tokens = nltk.tokenize.word_tokenize(s)

#         # remove short words, they're probably not useful
#         tokens = [t for t in tokens if len(t) >= self.min_length]

#         if self.lemmatize:
#             lemmatizer = WordNetLemmatizer()
#             # put words into base form
#             tokens = [lemmatizer.lemmatize(t) for t in tokens]

#         if self.stop_words:
#             # remove stopwords
#             tokens = [t for t in tokens if t not in self.stop_words]
#         if self.remove_digits:
#             # remove any digits, i.e. "3rd edition"
#             tokens = [t for t in tokens if not any(c.isdigit() for c in t)]

#         return tokens


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