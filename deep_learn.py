import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
import re
import nltk
from nltk.tag import pos_tag
import pandas as pd
import numpy as np
nltk.data.path.append('./nltk_data/')
from nltk.corpus import wordnet
import pickle
import validity_train

model_data = pickle.load(open('gib_model.pki', 'rb'))
model_mat = model_data['mat']
threshold = model_data['thresh']

def is_valid_string(document):
    valid =  validity_train.avg_transition_prob(document, model_mat) > threshold
    if valid:
        return(1)
    else:
        return(0)
    
def first_valid_word(document):
    strlst = document.split()
    if len(strlst) > 0:     
        valid =  validity_train.avg_transition_prob(strlst[0], model_mat) > threshold
        if valid:
            return(1)
        else:
            return(0)
    else:
        return(0)
    
def last_valid_word(document):
    strlst = document.split()
    if len(strlst) > 1:     
        valid =  validity_train.avg_transition_prob(strlst[len(strlst)-1], model_mat) > threshold
        if valid:
            return(1)
        else:
            return(0)
    else:
        return(0)    
    
def count_valid_words(document):
    strlst = document.split()
    word = 0
    for i in strlst:
        valid =  validity_train.avg_transition_prob(list(i), model_mat) > threshold
        if valid:
            word = word+1
    #print(word)        
    return(word)
   




# Count number of words in name
def count_words(document):
    return (len(document.split()))


# Check for 's in Name
def check_for_apos_s(document):
    document = re.findall(r'\b\'([s]\s)', document, flags=re.I)
    #print(str(document).find("'s "))
    #print(document)
    if(len(document) >= 1 ):
        return (1)
    else:
        return(0)
    
# Check for initials in Name
def has_initials(document):
    document = re.findall(r'\b\s([A-Z]\.\s)', document, flags=re.I)
    #print(str(document).find("'s "))
    #print(document)
    if(len(document) >= 1 ):
        return (1)
    else:
        return(0)    
    

    
def count_proper_noun(document):
    tagged_sent = pos_tag(document.split())
    #print(tagged_sent)
    propernouns = [pos for word,pos in tagged_sent if pos == 'NNP']
    return( len(propernouns))

def count_verbs(document):
    tagged_sent = pos_tag(document.split())
    #print(tagged_sent)
    propernouns = [pos for word,pos in tagged_sent if (pos == 'VB' or pos == 'VBG' or pos == 'VBD' or pos == 'VBN' or pos == 'VBZ' or pos       == 'VBP') ]
    return( len(propernouns))

def count_prepositions(document):
    tagged_sent = pos_tag(document.split())
    #print(tagged_sent)
    propernouns = [pos for word,pos in tagged_sent if pos == 'IN']
    return( len(propernouns))

def count_adjectives(document):
    tagged_sent = pos_tag(document.split())
    #print(tagged_sent)
    propernouns = [pos for word,pos in tagged_sent if (pos == 'JJ' or pos == 'JJR' or pos == 'JJS')]
    return( len(propernouns))

def count_personal(document):
    tagged_sent = pos_tag(document.split())
    #print(tagged_sent)
    propernouns = [pos for word,pos in tagged_sent if (pos == 'PRP' or pos == 'PRP$' )]
    return( len(propernouns))

def count_determiner(document):
    tagged_sent = pos_tag(document.split())
    propernouns = [pos for word,pos in tagged_sent if pos == 'DT']
    return( len(propernouns))

def count_cardinal_digit(document):
    tagged_sent = pos_tag(document.split())
    propernouns = [pos for word,pos in tagged_sent if (pos == 'CD')]
    return( len(propernouns))

def count_adverb(document):
    tagged_sent = pos_tag(document.split())
    propernouns = [pos for word,pos in tagged_sent if (pos == 'RP' or pos == 'RB' or pos == 'RBS' or pos == 'RBR')]
    return( len(propernouns))

def count_modal(document):
    tagged_sent = pos_tag(document.split())
    propernouns = [pos for word,pos in tagged_sent if (pos == 'MD')]
    return( len(propernouns))

def get_tag(document):
    tagged_sent = pos_tag(document.split())
    propernouns = [pos for word,pos in tagged_sent]
    return( propernouns)

def first_tag(document):
    strlst = document.split()
    if len(strlst) > 0:
        return( get_tag(strlst[0]))
    else:
        return(0)
    
def last_tag(document):
    strlst = document.split()
    if len(strlst) > 1:
        return( get_tag(strlst[len(strlst)-1]))
    else:
        return(0)
    
text = """
Alex T. L.
"""       
    

    
def get_features(df):
    newdf = pd.DataFrame()
    for i, row in df.iterrows():
        #print(df.iloc[[i]])
        dummy = df['id'][i]
        newdf.set_value(i,'id',dummy)
        
        dummy = df['is_person'][i]
        newdf.set_value(i,'is_person',dummy)

        dummy = df['str_of_words'][i]
        newdf.set_value(i,'str_of_words',dummy)
        
        str_of_words = df['str_of_words'][i]
        
        
        dummy = len(wordnet.synsets(str_of_words))
        newdf.set_value(i,'count_dict',dummy)
    
        strlst = str_of_words.split()
        if(len(strlst) > 0): 
            newdf.set_value(i,'first_dict',len(wordnet.synsets(strlst[0])))
        else:
            newdf.set_value(i,'first_dict',0)

        strlst = str_of_words.split()
        if(len(strlst) > 1): 
            newdf.set_value(i,'last_dict',len(wordnet.synsets(strlst[len(strlst)-1])))
        else:
            if(len(strlst) == 0):
                newdf.set_value(i,'last_dict',0)
            else:
                newdf.set_value(i,'last_dict',len(wordnet.synsets(strlst[0])))

        dummy = count_proper_noun(str_of_words)
        newdf.set_value(i,'nnp_count',dummy)
        #print("Number of Proper Nouns "+" =  "+str(dummy))

        dummy = count_verbs(str_of_words)
        newdf.set_value(i,'verb_count',dummy)
        #print("Number of verbs "+" =  "+str(dummy))

        dummy = count_prepositions(str_of_words)
        newdf.set_value(i,'prepos_count',dummy)
        #print("Number of Prepositions "+" =  "+str(dummy))

        dummy = count_adjectives(str_of_words)
        newdf.set_value(i,'adject_count',dummy)
        #print("Number of Adjectives "+" =  "+str(dummy))

        dummy = count_personal(str_of_words)
        newdf.set_value(i,'person_count',dummy)
        #print("Number of Personal and possesive pro nouns "+" =  "+str(dummy))

        dummy = count_determiner(str_of_words)
        newdf.set_value(i,'dt_count',dummy)
        #print("Number of Determiners "+" =  "+str(dummy))

        dummy = count_cardinal_digit(str_of_words)
        newdf.set_value(i,'cd_count',dummy)
        #print("Number of Cardinal Digit "+" =  "+str(dummy))

        dummy = count_adverb(str_of_words)
        newdf.set_value(i,'adverb_count',dummy)
        #print("Number of Adverbs "+" =  "+str(dummy))

        dummy = count_modal(str_of_words)
        newdf.set_value(i,'modal_count',dummy)
        #print("Number of Modals "+" =  "+str(dummy))

        dummy = check_for_apos_s(str_of_words)
        newdf.set_value(i,'has_apos',dummy)
        #print("Has apostrophe 's "+" =  "+str(dummy))                  

        dummy = first_tag(str_of_words)
        newdf.set_value(i,'first_name_tag',dummy)
        #print("NLTK toolkit tags first name as "+" =  "+str(dummy))

        dummy = first_tag(str_of_words.lower())
        newdf.set_value(i,'first_name_low_tag',dummy)
        #print("NLTK toolkit tags first name in lower case as "+" =  "+str(dummy))

        dummy = last_tag(text)
        newdf.set_value(i,'last_name_tag',dummy)
        #print("NLTK toolkit tags last name as "+" =  "+str(dummy))

        dummy = last_tag(str_of_words.lower())
        newdf.set_value(i,'last_name_low_tag',dummy)
        #print("NLTK toolkit tags last name in lower case as "+" =  "+str(dummy))

        dummy = has_initials(str_of_words)
        newdf.set_value(i,'has_initial',dummy)
        #print("The initials in the string "+" =  "+str(dummy))
        print(i)
        
        dummy = is_valid_string(str_of_words)
        newdf.set_value(i,'is_valid_string',dummy)
        
        dummy = count_valid_words(str_of_words)
        newdf.set_value(i,'count_valid_words',dummy)
        
        dummy = first_valid_word(str_of_words)
        newdf.set_value(i,'first_valid_word',dummy)
        
        dummy = last_valid_word(str_of_words)
        newdf.set_value(i,'last_valid_word',dummy)
        
            
        try:
            dummy = count_words(str_of_words)
            newdf.set_value(i,'word_count',dummy)
        except Exception as e:
            newdf.set_value(i,'word_count',False)
            pass
        #if(i>5):
            #break
        
    
    return newdf


h2o.init(max_mem_size = 4)
h2o.remove_all()

# load the model
saved_model = h2o.load_model('C:\idio\models\dl_v1')

str_of_words = input("Enter String : ")
df = pd.DataFrame()
df['id']=[1]
df['is_person']= [1]
df['str_of_words'] = [str(str_of_words)]
#print(df)
test = pd.DataFrame(get_features(df))
#print(test)



filename = 'datasets/dummy.csv'
test.to_csv(filename, index=False, encoding='utf-8')

test = pd.read_csv('./datasets/dummy.csv')
test['last_name_low_tag'] = test['last_name_low_tag'] == 0

test = h2o.H2OFrame(test)
pred = saved_model.predict(test[3:26]).as_data_frame(use_pandas=True)
print(pred['predict'].values)

#pred['predict'].values

h2o.cluster().shutdown(prompt=True)



