from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import numpy as np
import nltk
from nltk.tag import pos_tag
from nltk.corpus import brown
nltk.data.path.append('./nltk_data/')

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

def get_human_names():
    sparql.setQuery("""
    SELECT ?item ?itemLabel
    WHERE
    {
    ?item wdt:P31 wd:Q5.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    LIMIT 90000
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    df = pd.DataFrame(results_df[['itemLabel.value']])
    df = df.rename(columns={ 'itemLabel.value': 'str_of_words'})
    df['is_person'] = 1
    df['type'] = "human_name"
    filename = 'datasets\people.csv'
    df.to_csv(filename, index=False, encoding='utf-8')


def get_city_names():
    sparql.setQuery("""
        SELECT ?city ?cityLabel WHERE {
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        ?city wdt:P31 wd:Q515.
        }
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    #print(results_df)
    df = pd.DataFrame(results_df[[ 'cityLabel.value']])
    df = df.rename(columns={ 'cityLabel.value': 'str_of_words'})
    df['is_person'] = 0
    df['type'] = "city"
    filename = 'datasets\cities.csv'
    df.to_csv(filename, index=False, encoding='utf-8')

def get_country_names():
    sparql.setQuery("""
        SELECT ?city ?cityLabel WHERE {
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        ?city wdt:P31 wd:Q6256.
        }
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    #print(results_df)
    df = pd.DataFrame(results_df[[ 'cityLabel.value']])
    df = df.rename(columns={ 'cityLabel.value': 'str_of_words'})
    df['is_person'] = 0
    df['type'] = "country"
    filename = 'datasets\countries.csv'
    df.to_csv(filename, index=False, encoding='utf-8')

def get_region_names():
    sparql.setQuery("""
        SELECT ?city ?cityLabel WHERE {
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        ?city wdt:P31 wd:Q82794.
        }
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    #print(results_df)
    df = pd.DataFrame(results_df[[ 'cityLabel.value']])
    df = df.rename(columns={'cityLabel.value': 'str_of_words'})
    df['is_person'] = 0
    df['type'] = "geo_regions"
    filename = 'datasets\georegions.csv'
    df.to_csv(filename, index=False, encoding='utf-8')


def get_book_names():
    sparql.setQuery("""
        SELECT ?city ?cityLabel WHERE {
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        ?city wdt:P31 wd:Q571.
        }
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    #print(results_df)
    df = pd.DataFrame(results_df[['cityLabel.value']])
    df = df.rename(columns={ 'cityLabel.value': 'str_of_words'})
    df['is_person'] = 0
    df['type'] = "book_title"
    filename = 'datasets\gen_books.csv'
    df.to_csv(filename, index=False, encoding='utf-8')


def get_monarch():
    sparql.setQuery("""
        SELECT ?city ?cityLabel WHERE {
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        ?city wdt:P31 wd:Q116.
        }
        LIMIT 1000
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    #print(results_df)
    df = pd.DataFrame(results_df[['cityLabel.value']])
    df = df.rename(columns={'cityLabel.value': 'str_of_words'})
    df['is_person'] = 0
    df['type'] = "monarch"   
    filename = 'datasets\monarch.csv'
    df.to_csv(filename, index=False, encoding='utf-8')



    
def get_orgs():
    sparql.setQuery("""
    SELECT ?city ?cityLabel WHERE {
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    ?city wdt:P31 wd:Q43229.
    }
    LIMIT 50000        
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    #print(results_df)
    df = pd.DataFrame(results_df[['cityLabel.value']])
    df = df.rename(columns={'cityLabel.value': 'str_of_words'})
    df['is_person'] = 0
    df['type'] = "org"    
    filename = 'datasets\organizations.csv'
    df.to_csv(filename, index=False, encoding='utf-8')
    
    
    
def get_male_names():
    df = pd.read_csv('./datasets/White-Male-Names.csv')
    df = pd.DataFrame(df[[' first name', 'last name']])
    splits = df[' first name'].str.split()
    df['first'] = splits.str[0].str.title()
    df['middle'] = splits.str[1].str.title()
    df['last'] = df['last name'].str.title()
    df['middle'] = df["middle"].map(str)+'.'
    df['middle'] = df["middle"].str.replace('nan.', '')
    #Add required columns
    df['str_of_words'] = df["first"].map(str) +' '+df["middle"].map(str)+' '+df['last']
    df['is_person'] = 1
    df['type'] = 'human_name'
    #Drop  columns
    df = df.drop(' first name', 1)
    df = df.drop('last name', 1)
    df = df.drop('first', 1)
    df = df.drop('last', 1)
    df = df.drop('middle', 1)
    df = df.replace(np.nan, '', regex=True)
    filename = 'datasets\malenames.csv'
    df.to_csv(filename, index=False, encoding='utf-8')
    
    
def get_female_names():
    df = pd.read_csv('./datasets/White-Female-Names.csv')
    df = pd.DataFrame(df[[' first name', 'last name']])
    splits = df[' first name'].str.split()
    df['first'] = splits.str[0].str.title()
    df['middle'] = splits.str[1].str.title()
    df['last'] = df['last name'].str.title()
    df['middle'] = df["middle"].map(str)+'.'
    df['middle'] = df["middle"].str.replace('nan.', '')
    #Add required columns
    df['str_of_words'] = df["first"].map(str) +' '+df["middle"].map(str)+' '+df['last']
    df['is_person'] = 1
    df['type'] = 'human_name'
    #Drop  columns
    df = df.drop(' first name', 1)
    df = df.drop('last name', 1)
    df = df.drop('first', 1)
    df = df.drop('last', 1)
    df = df.drop('middle', 1)
    df = df.replace(np.nan, '', regex=True)
    filename = 'datasets\girlnames.csv'
    df.to_csv(filename, index=False, encoding='utf-8')
    
def get_hisp_female_names():
    df = pd.read_csv('./datasets/Hispanic-Female-Names.csv')
    df = pd.DataFrame(df[[' first name', 'last name']])
    splits = df[' first name'].str.split()
    df['first'] = splits.str[0].str.title()
    df['middle'] = splits.str[1].str.title()
    df['last'] = df['last name'].str.title()
    df['middle'] = df["middle"].map(str)+'.'
    df['middle'] = df["middle"].str.replace('nan.', '')
    #Add required columns
    df['str_of_words'] = df["first"].map(str) +' '+df["middle"].map(str)+' '+df['last']
    df['is_person'] = 1
    df['type'] = 'human_name'
    #Drop  columns
    df = df.drop(' first name', 1)
    df = df.drop('last name', 1)
    df = df.drop('first', 1)
    df = df.drop('last', 1)
    df = df.drop('middle', 1)
    df = df.replace(np.nan, '', regex=True)
    filename = 'datasets\hisp_girlnames.csv'
    df.to_csv(filename, index=False, encoding='utf-8')
    
def get_hisp_male_names():
    df = pd.read_csv('./datasets/Hispanic-Male-Names.csv')
    df = pd.DataFrame(df[['first name', 'last name']])
    splits = df['first name'].str.split()
    df['first'] = splits.str[0].str.title()
    df['middle'] = splits.str[1].str.title()
    df['last'] = df['last name'].str.title()
    df['middle'] = df["middle"].map(str)+'.'
    df['middle'] = df["middle"].str.replace('nan.', '')
    #Add required columns
    df['str_of_words'] = df["first"].map(str) +' '+df["middle"].map(str)+' '+df['last']
    df['is_person'] = 1
    df['type'] = 'human_name'
    #Drop  columns
    df = df.drop('first name', 1)
    df = df.drop('last name', 1)
    df = df.drop('first', 1)
    df = df.drop('last', 1)
    df = df.drop('middle', 1)
    df = df.replace(np.nan, '', regex=True)
    filename = 'datasets\hisp_malenames.csv'
    df.to_csv(filename, index=False, encoding='utf-8')
    
    

def get_noun(document):
    tagged_sent = pos_tag(document.split())
    words = []
    for word,pos in tagged_sent:
        if (pos == 'NN' and len(word) > 2):
            return( words)   
        
def get_nouns_from_text():
    df = pd.DataFrame()    
    news_sentences = brown.words(categories='news')
    things = []
    i = 1
    for w in news_sentences:
        if(get_noun(w.lower()) is not None):
             things = things+[str(w)]
        i = i+1
        print(i)
        if(i>400000):
            break
    df['str_of_words'] = things
    df['is_person'] = 0
    df['type'] = 'dictionary_words'
    filename = 'datasets\get_nouns.csv'
    df.to_csv(filename, index=False, encoding='utf-8') 
    #print(df)
    
if __name__ == '__main__':    
    #get_human_names()
    print("Got some Human names...")

    #get_city_names()
    print("Got some City names...")

    #get_country_names()
    print("Got all Country names...")

    #get_region_names()
    print("Got some Geo-Region names...")

    #get_book_names()
    print("Got some Book names...")

    #get_monarch()
    print("Got terms for Monarchy...")

    #get_orgs()
    print("Got some Organization names...")

    #get_male_names()
    print("Got some Male names...")

    #get_female_names()
    print("Got some Female names...")
    
    #get_hisp_male_names()
    print("Got some Hispanic Male names...")

    #get_hisp_female_names()
    print("Got some Hispanic Female names...")
    
    #get_nouns_from_text()
    print("Got nouns from text...")

    print(" All datasets downloded !!!")

