from Bio import Entrez
import datetime

def init(api_key, email):
    Entrez.api_key = api_key
    Entrez.email = email


def query_pubmed_by_date(date=None, retmax=10000):
    """
    Fetches Pubmed papers by date.
    :param date: Date in YYYY/MM/DD format. Defaults to yesterday.
    :param retmax: Number of records to fetch.
    :return: List of Pubmed IDs.
    """
    if date is None:
        date = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime('%Y/%m/%d')
    query = f'{date}[dp]'

    print(f"Querying Pubmed with: {query}")

    # fetch papers by term
    stream = Entrez.esearch(
        db="pubmed", term=query, retmax=retmax, usehistory="y"
    )
    record = Entrez.read(stream)
    return record["IdList"]

def query_pubmed(keywords_list, 
                 start_date="", 
                 end_date=datetime.datetime.today().strftime('%Y/%m/%d'), 
                 retmax=1000):
    """
    Fetches Pubmed papers by keywords and date.
    :param keywords_list: List of keywords to search for.
    :param start_date: Start date in YYYY/MM/DD format.
    :param end_date: End date in YYYY/MM/DD format. Defaults to today.
    :param retmax: Number of records to fetch.
    :return: List of Pubmed IDs.
    """
    query = " OR ".join(keywords_list)
    query = f"({query})"

    if start_date and end_date:
        query += f' AND ({start_date}:{end_date}[dp])'

    print(f"Querying Pubmed with: {query}")

    # fetch papers by term
    stream = Entrez.esearch(
        db="pubmed", term=query, retmax=retmax, usehistory="y"
    )
    record = Entrez.read(stream)
    return record["IdList"]



#adapted from https://stackoverflow.com/questions/73000220/extracting-multiple-abstracts-from-pubmed-from-external-pubmed-ids-list-with-bio
def fetch_abstracts(pub_ids,
                    chunksize=1000, 
                    output_file='abstracts.csv', 
                    append=False):
    """
    Fetches abstracts from Pubmed for a list of Pubmed IDs.
    :param pub_ids: List of Pubmed IDs.
    :param chunksize: Number of records to fetch at once.
    :param output_file: Output file name.
    :param append: Whether to append to the output file or overwrite it.
    :return: None
    """
    
    # Make sure requests to NCBI are not too big
    # Iterate with steps of `chunksize`
    for i in range(0, len(pub_ids), chunksize):
        j = i + chunksize
        if j >= len(pub_ids):
            j = len(pub_ids)

        print(f"Fetching abstracts: from {i} to {j}.")
        if isinstance(pub_ids[0], int):
            pub_ids = [str(x) for x in pub_ids]
        id_string = ','.join(pub_ids[i:j])
        #print(f'ID_STRING={id_string}')

        handle = Entrez.efetch(db="pubmed", id=id_string, rettype="xml", retmode="text", retmax=chunksize)
        
        records = Entrez.read(handle)
        mode = 'a' if append else 'w'
        with open(output_file, mode, newline='') as csvfile:
            csvfile.write(f'pmid\ttitle\tabstract\n')
            for pubmed_article in records['PubmedArticle']:
                #print(pubmed_article['MedlineCitation'].keys())
                pub_id = pubmed_article['MedlineCitation']['PMID']
                title = pubmed_article['MedlineCitation']['Article']['ArticleTitle']
                abstract = 'NA'
                if 'Abstract' in pubmed_article['MedlineCitation']['Article'].keys():
                    abstract = pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                #print(f'PUBMED_ID={pub_id}, TITLE={title}\n')
                csvfile.write(f'{pub_id}\t{title}\t{abstract}\n')


