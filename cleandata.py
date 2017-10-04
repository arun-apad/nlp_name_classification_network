import pandas as pd
import re
import unicodedata

if __name__ == '__main__':

    #df1 = pd.read_csv('./datasets/monarch.csv')
    #print(df1.head())
    df1 = pd.read_csv('./datasets/cities.csv')
    #print(df2.head())
    df3 = pd.read_csv('./datasets/countries.csv')
    #print(df3.head())
    #df4 = pd.read_csv('./datasets/georegions.csv')
    #print(df4.head())
    df5 = pd.read_csv('./datasets/gen_books.csv')
    #print(df5.head())
    df6 = pd.read_csv('./datasets/organizations.csv')
    #print(df6.head())
    df7 = pd.read_csv('./datasets/people.csv')
    #print(df7.head())
    df8 = pd.read_csv('./datasets/malenames.csv')
    #print(df7.head())
    df9 = pd.read_csv('./datasets/girlnames.csv')
    #print(df7.head())
    df10 = pd.read_csv('./datasets/gibberish.csv')
    #print(df7.head())
    df11 = pd.read_csv('./datasets/get_nouns.csv')
    #print(df7.head())
    df12 = pd.read_csv('./datasets/hisp_malenames.csv')
    #print(df7.head())
    df13 = pd.read_csv('./datasets/hisp_girlnames.csv')

    print("Datasets Loaded")



    df1 = df1.append(df3)
    df1 = df1.append(df5)
    df1 = df1.append(df6)
    df1 = df1.append(df7)
    df1 = df1.append(df8)
    df1 = df1.append(df9)
    df1 = df1.append(df10)
    df1 = df1.append(df11)
    df1 = df1.append(df12)
    df1 = df1.append(df13)

    # Free space
    del df3
    del df5
    del df6
    del df7
    del df8
    del df9
    del df10
    del df11
    del df12
    del df13

    #Drop type column, as we will not use it.
    df1 = df1.drop('type', 1)

    #Drop rows that the have Q123456 kind of values in column 'str_of_words'.
    pattern = '^Q[0-9]'
    df1 = df1[~df1['str_of_words'].str.contains(pattern)]
    pattern = '[0-9]'
    df1 = df1[~df1['str_of_words'].str.contains(pattern)]
    
    
    #Convert non-ascii charecter to ascii equivalent
    strwords = df1['str_of_words'].values
    df1['str_of_words'] = [unicodedata.normalize('NFKD', elem).encode('ascii', 'ignore') for elem in strwords]
    strwords = df1['str_of_words'].values
    df1['str_of_words'] = [ elem.decode('utf-8') for elem in strwords]

    #Adding an id column, for reference.
    df1['id'] = range(1, len(df1) + 1)

    #Get 33% of all data
    df1 = df1[df1['id']%3 == 0 ]

    filename = 'datasets/maindata.csv'
    df1.to_csv(filename, index=False, encoding='utf-8')

    print("Number of rows: "+str(len(df1)))
    print("Created our data, which is now ready for processing." )