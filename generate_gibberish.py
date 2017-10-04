import string
import random
import pandas as pd
from random import randint
import numpy as np


def low(size, disp):
    char_set =  string.ascii_lowercase
    return(''.join(random.sample(char_set*disp, size)))

def mix(size, disp):
    char_set = string.ascii_uppercase + string.ascii_lowercase
    return(''.join(random.sample(char_set*disp, size)))

def high(size, disp):
    char_set = string.ascii_uppercase 
    return(''.join(random.sample(char_set*disp, size)))

def digit(size, disp):
    char_set = string.ascii_uppercase + string.ascii_lowercase
    return(''.join(random.sample(char_set*disp, size)))
    
    
    
def get_gibberish():
    df = pd.DataFrame()
    newdf = pd.DataFrame()    
    for i in range(3000):
        jibs = low(randint(6, 15),randint(2, 8))
        newdf.set_value(i,'jibs',jibs)
    df = pd.DataFrame()
    for i in range(3000):
        jibs = mix(randint(6, 15),randint(2, 8))
        df.set_value(i,'jibs',jibs)
    newdf = newdf.append(df)
    df = pd.DataFrame()
    for i in range(3000):
        jibs = high(randint(6, 15),randint(2, 8))
        df.set_value(i,'jibs',jibs)                      
    newdf = newdf.append(df) 
    df = pd.DataFrame() 
    for i in range(3000):
        jibs = digit(randint(6, 15),randint(2, 8))
        df.set_value(i,'jibs',jibs)
        #print(count)
    newdf = newdf.append(df)
    return(newdf)
   
    
def get_gibberish_data():       
    newdf =  pd.DataFrame()     
    df =  get_gibberish()
    newdf['first'] = df['jibs']
    df =  get_gibberish()
    newdf['middle'] = df['jibs']
    df =  get_gibberish()
    newdf['last'] = df['jibs']   
    newdf["set1"] = newdf["first"]
    newdf["set2"] = newdf["first"].map(str) +" "+newdf["middle"]
    newdf["set3"] = newdf["first"].map(str) +" "+newdf["middle"]+" "+newdf['last']    
    newdf = newdf.drop('first', 1)
    newdf = newdf.drop('middle', 1)
    newdf = newdf.drop('last', 1)
    gibbs1 = np.asarray(newdf["set1"].values)
    gibbs2 = np.asarray(newdf["set2"].values)
    gibbs3 = np.asarray(newdf["set3"].values)
    gibbs1 = np.concatenate([gibbs1,gibbs2])
    gibbs1 = np.concatenate([gibbs1,gibbs3])
    return(gibbs1)

if __name__ == '__main__':
    gibbs = get_gibberish_data()
    df = pd.DataFrame()
    df['str_of_words'] = gibbs
    df['is_person'] = 0
    df['type'] = "gibberish"

    filename = 'datasets\gibberish.csv'
    df.to_csv(filename, index=False, encoding='utf-8')

    print("Generated our gibberish data, which is now ready for processing." )




    
