from Bio import Entrez
import datetime

def init(api_key, email):
    Entrez.api_key = api_key
    Entrez.email = email


#adapted from https://stackoverflow.com/questions/73000220/extracting-multiple-abstracts-from-pubmed-from-external-pubmed-ids-list-with-bio
def fetch_abstracts(pub_ids,
                    retmax=1000, 
                    output_file='abstracts.csv', 
                    append=False):
    # Make sure requests to NCBI are not too big
    # Iterate with steps of `retmax`
    for i in range(0, len(pub_ids), retmax):
        j = i + retmax
        if j >= len(pub_ids):
            j = len(pub_ids)

        print(f"Fetching abstracts from {i} to {j}.")
        id_string = ','.join(pub_ids[i:j])
        handle = Entrez.efetch(db="pubmed", id=id_string, rettype="xml", retmode="text", retmax=retmax)
        
        records = Entrez.read(handle)
        mode = 'a' if append else 'w'
        with open(output_file, mode, newline='') as csvfile:
            csvfile.write(f'mid\ttitle\tabstract\n')
            for pubmed_article in records['PubmedArticle']:
                #print(pubmed_article['MedlineCitation'].keys())
                pub_id = pubmed_article['MedlineCitation']['PMID']
                title = pubmed_article['MedlineCitation']['Article']['ArticleTitle']
                abstract = 'NA'
                if 'Abstract' in pubmed_article['MedlineCitation']['Article'].keys():
                    abstract = pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                #print(f'PUBMED_ID={pub_id}, TITLE={title}\n')
                csvfile.write(f'{pub_id}\t{title}\t{abstract}\n')

def query_pubmed(keywords_list, 
                 start_date="", 
                 end_date=datetime.datetime.today().strftime('%Y/%m/%d'), 
                 retmax=1000):
    
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


