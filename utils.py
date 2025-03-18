import pandas as pd
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import textwrap

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
# scientific units
from quantities import units
from wordcloud import WordCloud

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def read_abstract_data(set1_path, set2_path, delim="\t", text_labels = ["pest", "contr"]):
    '''reads and concatenates the two files with abstracts and adds labels'''
    set1 = pd.read_csv(set1_path, sep=delim)
    set2 = pd.read_csv(set2_path, sep=delim)
    print(set1.shape)
    print(set2.shape)
    set1['label'] = 1 # pesticides are the positive class
    set2['label'] = 0
    set1['text_label'] = text_labels[0]
    set2['text_label'] = text_labels[1]
    #print(set2.head())
    #concatenate and reset Index
    data = pd.concat([set1, set2], axis=0)
    data = data.reset_index(drop=True)

    data = data.dropna()
    return data

def preprocess_text(data, keep_original=True, remove_punct=False, lower_case=True):
    '''preprocesses the text data'''
    if remove_punct :
        #remove punctuation
        punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~'   # `|` is not present here
        transtab = str.maketrans(dict.fromkeys(punct, ''))
        if keep_original:            
            data['abstract_clean'] = data['abstract'].str.translate(transtab)
            data['title_clean'] = data['title'].str.translate(transtab)
        else:
            data['abstract'] = data['abstract'].str.translate(transtab)
            data['title'] = data['title'].str.translate(transtab)
    if lower_case:
        ## lowercasing
        if keep_original:            
            data['title_clean'] = data['title'].str.lower()
            data['abstract_clean'] = data['abstract'].str.lower()
        else:
            data['title'] = data['title'].str.lower()
            data['abstract'] = data['abstract'].str.lower()
    #return None

def get_stopwords(extended=True, add_scientific_units=True, custom=None):
    '''constructs a set of stopwords and returns this as a list'''
    stops = set(stopwords.words('english'))
    if extended:
        stops = stops.union({
            'said', 'would', 'could', 'told', 'also', 'one', 'two',
            'mr', 'new', 'year', 'used'
        })
    if add_scientific_units:
        unit_symbols = [u.symbol for _, u in units.__dict__.items() if isinstance(u, type(units.deg))]
        ## add scientific units to stopwords
        stops = stops.union(unit_symbols)
    if custom:
        stops = stops.union(custom)
    stops = list(stops)
    return stops

class StemTokenizer:
    def __init__(self):
        self.porter = PorterStemmer()
    def __call__(self, doc):
        tokens = word_tokenize(doc)
        return [self.porter.stem(t) for t in tokens]
    
class Tokenizer:
    '''class for tokenizing text data'''
    def __init__(self, min_length=2, lemmatize=True, stop_words=get_stopwords(), remove_digits=True):
        self.min_length = min_length
        self.lemmatize = lemmatize
        self.stop_words = stop_words
        self.remove_digits = remove_digits

    def __call__(self, s):
        return self.tokenize_text(s)
    
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


def plot_top_words(model, feature_names, n_components=10, n_top_words=10):
    '''plots the top words for each topic'''
    import math
    rows = 2
    cols = math.ceil(n_components / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
            fig.suptitle('Topic Top Words', fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def evaluate_model(model, x_train, y_train, x_test, y_test, verbose = True):
    '''evaluates the already fitted model and prints the results'''
    # print('======')
    # print(y_test[1:10])
    # print('======')
    # print(y_train[1:10])
    # print('======')
    y_pred = model.predict(x_test)

    results = {}

    num_digits = 5
    train_accuracy = model.score(x_train, y_train)
    results['train_accuracy'] = train_accuracy

    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    results['accuracy'] = test_accuracy

    test_precision = metrics.precision_score(y_test, y_pred)
    results['precision'] = test_precision

    test_recall = metrics.recall_score(y_test, y_pred)
    results['recall'] = test_recall

    test_f1 = metrics.f1_score(y_test, y_pred)
    results['f1'] = test_f1

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cnf_matrix

    if verbose:
        print("test accuracy: ", round(test_accuracy, num_digits))
        print("train accuracy:", round(train_accuracy, num_digits))
        print("test precision:", round(test_precision, num_digits))
        print("test recall:   ", round(test_recall, num_digits))
        print("test F1 score: ", round(test_f1, num_digits))
        print("confusion matrix:")
        print(cnf_matrix)
    return results

    
# Scikit-Learn is transitioning to V1 (and can plot this itself),
# see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
# but it's not available on Colab yet 
# The changes modify how confusion matrices are plotted
def plot_confusion_matrix(cm, class_labels):
    plt.figure(figsize=(4, 3))
    #sn.set_theme(rc={'figure.figsize':(7, 5)}) #(11.7,8.27)
    df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    ax = sn.heatmap(df_cm, annot=True, fmt='g')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")

def visualize(df, label):
    '''visualizes the data with a wordcloud.
    
    Arguments:
        df = DataFrame with the data
        label = label to visualize'''
    words = ''
    for msg in df[df['text_label'] == label]['abstract_clean']:
        words += msg + ' '
    make_wordcloud(words)


def make_wordcloud(words):
    '''Takes a string of space-separated words to generatye a wordcloud'''
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


def sample_validation_set(positives_file, negatives_file, 
                          n_positive=30, n_negative=470,
                          out_file='validation_set.csv',
                          abstract_required=True):
    '''Samples a validation set from positive and negative samples'''

    # read positive and negative samples
    #print(f'positives={positives_file}, negatives={negatives_file}')

    positives = pd.read_csv(positives_file, sep='\t', na_values="NA")
    negatives = pd.read_csv(negatives_file, sep='\t', na_values="NA")

    if abstract_required:
        # filter out rows without abstract
        positives = positives[positives['abstract'].notna()]
        negatives = negatives[negatives['abstract'].notna()]

    # sample validation set
    validation_positives = positives.sample(n=n_positive)
    validation_negatives = negatives.sample(n=n_negative)

    # add label
    validation_positives['label'] = 1
    validation_negatives['label'] = 0
    validation_positives['text_label'] = 'pest'
    validation_negatives['text_label'] = 'contr'

    #concatenate
    validation_set = pd.concat([validation_positives, validation_negatives])

    #remove duplicates (in set pest and set contr)
    validation_set = validation_set.drop_duplicates()

    # shuffle
    validation_set = validation_set.sample(frac=1)

    # write validation set
    validation_set.to_csv(out_file, sep='\t', index=False)

def print_paper(pmid, df, width = 80):
    '''Prints a paper in readable format.

    Arguments:
        pmid = pubmed-id
        df = DataFrame with papers
        width = width of the text
    '''
    record = df[df['pmid']==pmid]
    #print(record)
    print(f'PubMed ID: {pmid}')
    print(f'\nTitle: ')
    print("\n".join(textwrap.wrap(record.iloc[0,1], width = width)))
    print(f'\nAbstract:')
    print("\n".join(textwrap.wrap(record.iloc[0,2], width = width)))

    # create a function that generates recommendations
def recommend(pmid, df, paper2idx, X, top_n_papers = 5):
    ''' Recommends similar papers based on the cosine similarity of the TF-IDF matrix.

    Arguments:
        pmid: pubmed-id
        df: DataFrame with papers that was used to build X
        paper2idx: mapping of pubmedid to dataframe index
        X: TF-IDF sparse matrix
        top_n_papers: number of recommendations to return (default = 5)
    '''
    # will skip the first (self hit)
    top_n_papers += 1
    # get the row in the dataframe for this movie
    idx = paper2idx[pmid]
    if type(idx) == pd.Series: # when there are multiple mvies with the same title
        idx = idx.iloc[0]
    
    # calculate the pairwise similarities for this movie
    query = X[idx]
    scores = cosine_similarity(query, X)
    
    # currently the array is 1 x N, make it just a 1-D array
    scores = scores.flatten()
    
    # get the indexes of the highest scoring movies
    # get the first K recommendations
    # don't return itself!
    top_scoring_idx = (-scores).argsort()[1:top_n_papers]
    top_scoring_scores = scores[top_scoring_idx]

    best_matches_pmids = df['pmid'].iloc[top_scoring_idx] # fetches a Series
    best_matches_titles = df['title'].iloc[top_scoring_idx]    
    best_matches = pd.concat([best_matches_pmids, best_matches_titles], axis=1)
    best_matches['score'] = top_scoring_scores
    return best_matches