##To classify whether given string is the name of a Person.

Contents

1. Creating a data corpus.
  - Identifying data resources.
  - Getting the data.
  - Cleaning and filtering the data.
2. Modelling features of a person&#39;s name.
  - Detect and engineer validity of string features.
  - Detect and engineer parts of speech features.
  - Detect and engineer dictionary features.
3. Name detection using Machine Learning.
  - Deep Learning using H2O.ai
  - Predicting using H2O.ai
4. Step by step Instructions.
  - Run as is.
  - Run from scratch.



### 1. Creating a data corpus.

**Identifying Data resources:**

The idea is to have a corpus of data with labelled person names and non-person namesto work with. The following were identified as resources for person and non-person names.

- **DBpedia&#39; Wikidata:** [https://www.wikidata.org/wiki/Wikidata:SPARQL\_query\_service/queries/examples](https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples)  For Person and non-person names.
- **NLTK&#39;s text corpora:** [http://www.nltk.org/book/ch02.html](http://www.nltk.org/book/ch02.html) For non-person names.
- **Data generation** : Generated some data programmatically for non-person names.
- **Male and Female names from git** : [https://mbejda.github.io/](https://mbejda.github.io/) For person names.

In the following task I have not used other methods such as web crawling to get names and other data sources for names such as:

[https://www.drupal.org/project/name](https://www.drupal.org/project/name)

[http://www.quietaffiliate.com/free-first-name-and-last-name-databases-csv-and-sql/](http://www.quietaffiliate.com/free-first-name-and-last-name-databases-csv-and-sql/)

[http://www.namepedia.org/en/firstname/](http://www.namepedia.org/en/firstname/)

**Getting and cleaning the data:**

Once the data sources were identified, the next step was to get the data.  While getting the data, all Person/People names were tagged with &quot;is\_person = 1&quot; and rest were tagged with &quot;is\_person = 0&quot;. All the code required for getting the data are in **getdata.py** and **generate\_gibberish.py** python files. All data files will be downloaded to the **./datasets** folder within the project folder. Below are the methods used to get data from respective data sources.

- **DBpedia Wikidata:** Queried the DBpedia&#39;s wikiData using a python library called SPARQLWrapper. The library allows us to query wiki data using a query language called SPARQL. Collected **Person names** , **country names** (person names have rarely country names in them), **city names** (person names have rarely city names in them), **region names** (person names sometimes do have region names in them), **book names** (person names rarely have non nouns in them), **organization names** (person names have rarely organization like names in them) and **names of monarchs** (person names may have titles of monarchy in them). The idea was to get data for which features of a person&#39;s name can be modelled. All the functions to get each data is available in the **getdata.py** python file. Around 80,000 person names and 100,000 non-person names were downloaded from wikidata.
- **NLTK&#39;s text corpora:**  Used the python command line tool to download corpus. Wrote a function to mine words tagged as common nouns from one of the text corpora. The function that does this is called **get\_noun()** in **getdata.py.** Around 15,000 non-person names were mined from the text corpus.
- **Data generation** : Wrote a function to generate gibberish data to be added in the mix. The function is in **generate\_gibberish.py** python file. Around 10,000 gibberish non-person names were generated.
- **Male and Female names from git** : Manually unzipped the csv collections of names from [https://mbejda.github.io/](https://mbejda.github.io/) . Have used Caucasian male-females and Hispanic male-female names. Also included a large text of articles for training Markov model to identify a word as a valid word or gibberish. It can be found in the **./datasets** folder as **text\_collections.txt**.

**Cleaning and filtering the data:**

All the data that was downloaded from previous step is located in the **./datasets**. The next step is to clean and filter this data that was gathered from different sources to a main data file for the project. This file is called the maindata.csv and is available in the **./datasets** folder. Below are the measures taken to clean and filter the data and is available in **cleandata.py**.

- Removed some bad data that came along with wikidata, values such as &#39;Qbbsuhfoew87&#39; etc.
- Converted non-ascii characters to their asci equivalent.
- Added an &quot;id&quot; column to the data for referencing.
- Filtered 33% of the 300,000+ rows of person and non-person names as the main data to be used in solving the task.
- Saved the data to a csv file named **maindata.csv**.
- The maindata.csv now has the following columns: **id, str\_of\_words, is\_person**. The **str\_of\_words** column holds both person and non-person names. The **is\_person** column holds the values 1 for person names and 0 for non-person names respectively.
- Please note that I have not added region names and names of monarch to the main data. These can be added to main data if required in a future enhancement.

### 2. Modelling features of a person&#39;s name.

The idea here is to model person and non-person characteristics in a given string. The features are engineered for the 100000+ person and non-person names in **maindata.csv**. The features are engineered and saved to **pos\_dictionary\_features.csv**. All the code for the feature engineering is available in **pos\_dictionary\_validity\_features.py**. Primarily used 3 methods to engineer feature of a person&#39;s name.

**Detect and engineer validity of string features:**

Trained the **text\_collections.txt** with a Markov model to predict if a given string is a valid word or just gibberish. The code to train the model can be found in the **validity\_train.py** file. Have used the prediction code in **pos\_dictionary\_validity.py** to model features such as:

**Count\_valid\_words:** Counts the number of valid words in the given string (int).

**is\_valid\_word:** Are all the words of the given string valid (1/0).

**first\_valid\_word:** Is the first word of the given string is valid (1/0).

**last\_valid\_word:** Is the last word of the given string is valid (1/0).

**Detect and engineer parts of speech features:**

Used the nltk library in python to detect parts of speech such as verb, adjectives, proper nouns, etc. Have used the library to engineer parts of speech features such as:

**nnp\_count:** Count the number of Proper Nouns in the given string (int).

**verb\_count:** Count the number of Verbs in the given string. Here it counts all forms of verbs(past, present, future and future tense) (int).

**prepos\_count:** Counts the number of prepositions (at, in, on..) in the given string (int).

**adject\_count:** Counts the number of adjectives (big, small, red..) in the given string (int).

**person\_count:** Counts the number of possessive pronouns (his, her, they..) in the given string (int).

**dt\_count:** Counts the number of determiners (a, an, the..) in the given string (int).

**cd\_count:** Counts the number of words with number meanings (one, two, three..) in the given string (int).

**adverb\_count:** Counts the number of adverbs (here, now, very..) in the given string.

**modal\_count:** Counts the number of modals (should, would..) in the given string (int).

**has\_apos:** Does the string have an apostrophe s (&#39;s) in it(1,0).

**has\_initial:** Does the string have initials (Jake T. Ralf) in it(1,0).

**first\_tag:** What is the first word tagged as in the give string (Proper Noun, verb, adverb..) (category).

**last\_tag:** What is the first word tagged as in the give string (Proper Noun, verb, adverb..) (category).

**first\_name\_low\_tag:** What is the first word tagged as in the give string when all in lower case (Proper Noun, verb, adverb..) (category).

**last\_name\_low\_tag:** What is the last word tagged as in the give string when all in lower case (Proper Noun, verb, adverb..) (category).

**count\_word** : counts the number of words in the string (int).

There other parts of speech that can be used based on the context, find full list at [http://www.ling.upenn.edu/courses/Fall\_2003/ling001/penn\_treebank\_pos.html](http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

**Detect and engineer dictionary features:**

Use the **wordnet** library in nltk corpus to identify words with meanings.  Have used the number of synonyms in this case.

**count\_dict:** Count the number of synonyms in the dictionary for all the words in the given string (int).

**first\_dict:** Count the number of synonyms in the dictionary for the first word in the given string (int).

**last\_dict:** Count the number of synonyms in the dictionary for the last word in the given string (int).

There are other counts we can use such as antonyms, etc but left as a future exercise. Also include code to check for the meaning of word on google and on the data on Priceton&#39;s website for wordnet, but have used the nltk corpus instead.

### 3. Name detection using Machine Learning.

**Deep Learning using H2O.ai:**

Once the features are created in pos\_dictionary.csv, it will be used as input to the deep learning model built on h2o.ai framework. The h2o.ai is a scalable framework for Machine learning and neural networks that are designed to run on distributed clusters. [https://www.h2o.ai/](https://www.h2o.ai/). All the code related to training the DL model is in **train\_deep.py**. Please note that the framework will initialize a single node cluster on the machine every time we run it. Ideally there would be cluster already running on which this should be executed.

- From **pos\_dictionary\_validity.csv** we split the data into 3 part for training(60,000), testing(20,000) and validation(20,000).
- We assign the is\_person column as the dependent variable and the rest as independent variables. We omit the id and str\_of\_words columns as str\_of\_words will be the column which will be our input during prediction.
- A deep learning and validation is done on the train and validation sets. This was then predicted on an unseen test set.
- The model was tuned to get accuracy as follows:

Was person name and predicted as True   96.1343669250646
Was not person name but predicted as True   12.016699945543657
Was person name but predicted as False   3.8656330749354004
Was not person name and predicted as False   87.98330005445635

- The model is saved into **./models** folder for use in the next step, prediction.

- The framework will initialize a cluster of 1 node in your local machine local machine to run the training.

**Predicting using H2O.ai:**

The saved model can now be used to predict if a string is a person&#39;s name or not. The deep\_learn.py has all code to run the prediction.

- The program will take the input string and model the feature and then use the saved model to predict.
- The output will be a true or false value.

#### 4. Step by Step instructions

**1. Run as is.**   

* Copy folder to location on local machine.  
* Install all libraries.

pip install pandas  
pip install SPARQLWrapper  
pip istall numpy  
pip install nltk  
pip install unicodedata  
pip install pickle  
pip install h2o 
(Note the h2o requires java in the local machine) 

* Run the prediction file  
 python deep_learn.py  

* A h2o session will be created and you be asked to enter the string. 
* Enter the string to detect person name or not.
* Output will either be True or False.


** 1. Run from scratch.**   
* Copy folder to location on local machine.  
* Install all libraries.

pip install pandas  
pip install SPARQLWrapper  
pip istall numpy  
pip install nltk  
pip install unicodedata  
pip install pickle  
pip install h2o 
(Note the h2o requires java in the local machine)

* Get the data from nltk.  
python -m nltk.downloader

This will open a pop up from which you can download the file. Make sure to point the location to ./nltk_data in your project folder.

* Get White male and female names and Hispanic male and female names data manually from [https://mbejda.github.io/](https://mbejda.github.io/) and place in ./datasets folder within the project folder. Get the text_collection.txt (or use a large 6MB text file) into the ./datasets folder, to be used by validity_train.py  

* Run the getdata.py which will get all the other data required for our task.  

python getdata.py  

Alternatively just copy paste the datasets from the given folder to avoid waiting for data to be downloaded.

* Run the generate_gibberish.py to get some gibberish data into ./datasets folder

python generate_gibberish.py

* Run the cleandata.py to create our maindata.csv.  

python cleandata.py

* Run the validity_train.py which will save a markov model that will be used in the next step.

python validity_train.py

* Run the pos_dictionary_validity_features.py to do the feature engineering. (Takes about 30 mins for 100,000 rows). This will create pos_dictionary_features.csv by using the saved markov model, parts of speech  and dictionary features. This file uses the maindata.csv previously created. 

python pos_dictionary_validity_features.py

This will create pos_dictionary_validity.csv in ./datasets folder.

* Run the train_deep.py for training the featured data from pos_dictionary_validity.csv

python train_deep.py

Initially this will setup the h2o cluster and do the training. The output here shows the accuracy of the model like this.

Was person name and predicted as True   91.7829457364341
Was not person name but predicted as True   7.705572699219459
Was person name but predicted as False   8.217054263565892
Was not person name and predicted as False   92.29442730078054

* Run the prediction file  
 python deep_learn.py  

* A h2o session will be created and you be asked to enter the string. 
* Enter the string to detect person name or not.
* Output will either be True or False.



##To classify whether given string is the name of a Person.

Contents

1. Creating a data corpus.
  - Identifying data resources.
  - Getting the data.
  - Cleaning and filtering the data.
2. Modelling features of a person&#39;s name.
  - Detect and engineer validity of string features.
  - Detect and engineer parts of speech features.
  - Detect and engineer dictionary features.
3. Name detection using Machine Learning.
  - Deep Learning using H2O.ai
  - Predicting using H2O.ai
4. Step by step Instructions.
  - Run as is.
  - Run from scratch.



### 1. Creating a data corpus.

**Identifying Data resources:**

The idea is to have a corpus of data with labelled person names and non-person namesto work with. The following were identified as resources for person and non-person names.

- **DBpedia&#39; Wikidata:** [https://www.wikidata.org/wiki/Wikidata:SPARQL\_query\_service/queries/examples](https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples)  For Person and non-person names.
- **NLTK&#39;s text corpora:** [http://www.nltk.org/book/ch02.html](http://www.nltk.org/book/ch02.html) For non-person names.
- **Data generation** : Generated some data programmatically for non-person names.
- **Male and Female names from git** : [https://mbejda.github.io/](https://mbejda.github.io/) For person names.

In the following task I have not used other methods such as web crawling to get names and other data sources for names such as:

[https://www.drupal.org/project/name](https://www.drupal.org/project/name)

[http://www.quietaffiliate.com/free-first-name-and-last-name-databases-csv-and-sql/](http://www.quietaffiliate.com/free-first-name-and-last-name-databases-csv-and-sql/)

[http://www.namepedia.org/en/firstname/](http://www.namepedia.org/en/firstname/)

**Getting and cleaning the data:**

Once the data sources were identified, the next step was to get the data.  While getting the data, all Person/People names were tagged with &quot;is\_person = 1&quot; and rest were tagged with &quot;is\_person = 0&quot;. All the code required for getting the data are in **getdata.py** and **generate\_gibberish.py** python files. All data files will be downloaded to the **./datasets** folder within the project folder. Below are the methods used to get data from respective data sources.

- **DBpedia Wikidata:** Queried the DBpedia&#39;s wikiData using a python library called SPARQLWrapper. The library allows us to query wiki data using a query language called SPARQL. Collected **Person names** , **country names** (person names have rarely country names in them), **city names** (person names have rarely city names in them), **region names** (person names sometimes do have region names in them), **book names** (person names rarely have non nouns in them), **organization names** (person names have rarely organization like names in them) and **names of monarchs** (person names may have titles of monarchy in them). The idea was to get data for which features of a person&#39;s name can be modelled. All the functions to get each data is available in the **getdata.py** python file. Around 80,000 person names and 100,000 non-person names were downloaded from wikidata.
- **NLTK&#39;s text corpora:**  Used the python command line tool to download corpus. Wrote a function to mine words tagged as common nouns from one of the text corpora. The function that does this is called **get\_noun()** in **getdata.py.** Around 15,000 non-person names were mined from the text corpus.
- **Data generation** : Wrote a function to generate gibberish data to be added in the mix. The function is in **generate\_gibberish.py** python file. Around 10,000 gibberish non-person names were generated.
- **Male and Female names from git** : Manually unzipped the csv collections of names from [https://mbejda.github.io/](https://mbejda.github.io/) . Have used Caucasian male-females and Hispanic male-female names. Also included a large text of articles for training Markov model to identify a word as a valid word or gibberish. It can be found in the **./datasets** folder as **text\_collections.txt**.

**Cleaning and filtering the data:**

All the data that was downloaded from previous step is located in the **./datasets**. The next step is to clean and filter this data that was gathered from different sources to a main data file for the project. This file is called the maindata.csv and is available in the **./datasets** folder. Below are the measures taken to clean and filter the data and is available in **cleandata.py**.

- Removed some bad data that came along with wikidata, values such as &#39;Qbbsuhfoew87&#39; etc.
- Converted non-ascii characters to their asci equivalent.
- Added an &quot;id&quot; column to the data for referencing.
- Filtered 33% of the 300,000+ rows of person and non-person names as the main data to be used in solving the task.
- Saved the data to a csv file named **maindata.csv**.
- The maindata.csv now has the following columns: **id, str\_of\_words, is\_person**. The **str\_of\_words** column holds both person and non-person names. The **is\_person** column holds the values 1 for person names and 0 for non-person names respectively.
- Please note that I have not added region names and names of monarch to the main data. These can be added to main data if required in a future enhancement.

### 2. Modelling features of a person&#39;s name.

The idea here is to model person and non-person characteristics in a given string. The features are engineered for the 100000+ person and non-person names in **maindata.csv**. The features are engineered and saved to **pos\_dictionary\_features.csv**. All the code for the feature engineering is available in **pos\_dictionary\_validity\_features.py**. Primarily used 3 methods to engineer feature of a person&#39;s name.

**Detect and engineer validity of string features:**

Trained the **text\_collections.txt** with a Markov model to predict if a given string is a valid word or just gibberish. The code to train the model can be found in the **validity\_train.py** file. Have used the prediction code in **pos\_dictionary\_validity.py** to model features such as:

**Count\_valid\_words:** Counts the number of valid words in the given string (int).

**is\_valid\_word:** Are all the words of the given string valid (1/0).

**first\_valid\_word:** Is the first word of the given string is valid (1/0).

**last\_valid\_word:** Is the last word of the given string is valid (1/0).

**Detect and engineer parts of speech features:**

Used the nltk library in python to detect parts of speech such as verb, adjectives, proper nouns, etc. Have used the library to engineer parts of speech features such as:

**nnp\_count:** Count the number of Proper Nouns in the given string (int).

**verb\_count:** Count the number of Verbs in the given string. Here it counts all forms of verbs(past, present, future and future tense) (int).

**prepos\_count:** Counts the number of prepositions (at, in, on..) in the given string (int).

**adject\_count:** Counts the number of adjectives (big, small, red..) in the given string (int).

**person\_count:** Counts the number of possessive pronouns (his, her, they..) in the given string (int).

**dt\_count:** Counts the number of determiners (a, an, the..) in the given string (int).

**cd\_count:** Counts the number of words with number meanings (one, two, three..) in the given string (int).

**adverb\_count:** Counts the number of adverbs (here, now, very..) in the given string.

**modal\_count:** Counts the number of modals (should, would..) in the given string (int).

**has\_apos:** Does the string have an apostrophe s (&#39;s) in it(1,0).

**has\_initial:** Does the string have initials (Jake T. Ralf) in it(1,0).

**first\_tag:** What is the first word tagged as in the give string (Proper Noun, verb, adverb..) (category).

**last\_tag:** What is the first word tagged as in the give string (Proper Noun, verb, adverb..) (category).

**first\_name\_low\_tag:** What is the first word tagged as in the give string when all in lower case (Proper Noun, verb, adverb..) (category).

**last\_name\_low\_tag:** What is the last word tagged as in the give string when all in lower case (Proper Noun, verb, adverb..) (category).

**count\_word** : counts the number of words in the string (int).

There other parts of speech that can be used based on the context, find full list at [http://www.ling.upenn.edu/courses/Fall\_2003/ling001/penn\_treebank\_pos.html](http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

**Detect and engineer dictionary features:**

Use the **wordnet** library in nltk corpus to identify words with meanings.  Have used the number of synonyms in this case.

**count\_dict:** Count the number of synonyms in the dictionary for all the words in the given string (int).

**first\_dict:** Count the number of synonyms in the dictionary for the first word in the given string (int).

**last\_dict:** Count the number of synonyms in the dictionary for the last word in the given string (int).

There are other counts we can use such as antonyms, etc but left as a future exercise. Also include code to check for the meaning of word on google and on the data on Priceton&#39;s website for wordnet, but have used the nltk corpus instead.

### 3. Name detection using Machine Learning.

**Deep Learning using H2O.ai:**

Once the features are created in pos\_dictionary.csv, it will be used as input to the deep learning model built on h2o.ai framework. The h2o.ai is a scalable framework for Machine learning and neural networks that are designed to run on distributed clusters. [https://www.h2o.ai/](https://www.h2o.ai/). All the code related to training the DL model is in **train\_deep.py**. Please note that the framework will initialize a single node cluster on the machine every time we run it. Ideally there would be cluster already running on which this should be executed.

- From **pos\_dictionary\_validity.csv** we split the data into 3 part for training(60,000), testing(20,000) and validation(20,000).
- We assign the is\_person column as the dependent variable and the rest as independent variables. We omit the id and str\_of\_words columns as str\_of\_words will be the column which will be our input during prediction.
- A deep learning and validation is done on the train and validation sets.
- The model was tuned to get accuracy as follows:  

Was person name and predicted as True   91.7829457364341
Was not person name but predicted as True   7.705572699219459
Was person name but predicted as False   8.217054263565892
Was not person name and predicted as False   92.29442730078054  

- The model is saved into **./models** folder for use in the next step, prediction.

- The framework will initialize a cluster of 1 node in your local machine local machine to run the training.

**Predicting using H2O.ai:**

The saved model can now be used to predict if a string is a person&#39;s name or not. The deep\_learn.py has all code to run the prediction.

- The program will take the input string and model the feature and then use the saved model to predict.
- The output will be a true or false value.

#### 4. Step by Step instructions

**1. Run as is.**   

* Copy folder to location on local machine.  
* Install all libraries.

pip install pandas  
pip install SPARQLWrapper  
pip istall numpy  
pip install nltk  
pip install unicodedata  
pip install pickle  
pip install h2o 
(Note the h2o requires java in the local machine) 

* Run the prediction file  
 python deep_learn.py  

* A h2o session will be created and you will be asked to enter the string. 
* Enter the string to detect person name or not.
* Output will either be True or False.


**2. Run from scratch.**  
  
* Copy folder to location on local machine.  
* Install all libraries.

pip install pandas  
pip install SPARQLWrapper  
pip istall numpy  
pip install nltk  
pip install unicodedata  
pip install pickle  
pip install h2o 
(Note the h2o requires java in the local machine)

* Get the data from nltk.  
python -m nltk.downloader

This will open a pop up from which you can download the file. Make sure to point the location to ./nltk_data in your project folder.

* Get White male and female names and Hispanic male and female names data manually from [https://mbejda.github.io/](https://mbejda.github.io/) and place in ./datasets folder within the project folder. Get the text_collection.txt (or use a large 6MB text file) into the ./datasets folder, to be used by validity_train.py  

* Run the getdata.py which will get all the other data required for our task.  

python getdata.py  

Alternatively just copy paste the datasets from the given folder to avoid waiting for data to be downloaded.

* Run the generate_gibberish.py to get some gibberish data into ./datasets folder

python generate_gibberish.py

* Run the cleandata.py to create our maindata.csv.  

python cleandata.py

* Run the validity_train.py which will save a markov model that will be used in the next step.

python validity_train.py

* Run the pos_dictionary_validity_features.py to do the feature engineering. (Takes about 30 mins for 100,000 rows). This will create pos_dictionary_features.csv by using the saved markov model, parts of speech  and dictionary features. This file uses the maindata.csv previously created. 

python pos_dictionary_validity_features.py

This will create pos_dictionary_validity.csv in ./datasets folder.

* Run the train_deep.py for training the featured data from pos_dictionary_validity.csv

python train_deep.py

Initially this will setup the h2o cluster and do the training. The output here shows the accuracy of the model like this.

Was person name and predicted as True   96.1343669250646
Was not person name but predicted as True   12.016699945543657
Was person name but predicted as False   3.8656330749354004
Was not person name and predicted as False   87.98330005445635

Also look at closely on results by uncommenting line 78 in train_deep.py  

* Run the prediction file  
 python deep_learn.py  

* A h2o session will be created and you be asked to enter the string. 
* Enter the string to detect person name or not.
* Output will either be True or False.


### Notes:

Output correctness:

* The model does well most often in getting the name as true and non-name as false. Seems to be guessing American names better, maybe cause of the data it was trained with.  
* The model does well in diffentiating objects from person names.
* The model does well in diffentiating gibberish from person name.

##Examples of 20 for each case

* Was person name and predicted as True

Corazon Aquino  
Nicolas Sarkozy  
Louis IX of France  
Carl Sagan  
  Eduardo Frei Montalva  
        Frederic Taddei  
     Andre-Marie Ampere  
    Pedro Aguirre Cerda  
        Fabrizio Donato  
           Patrice Evra  
          Djibril Cisse  
            Donald Tusk  
                U Thant  
             Trygve Lie  
          German Riesco  
 Manuel Blanco Encalada  
         Vladimir Lenin  
           Hans Schmidt  
        Konrad Adenauer  
        Friedrich Ebert  

* Was not person name but predicted as True

 Nowy Sacz  
                Nizhny Novgorod  
                        Guatemala City  
                             Siem Reap  
                            Glen Ullin  
          Shibin Al Kawm, Al Minufiyah  
                      Charlotte Amalie  
                    Sroda Wielkopolska  
                             Cave City  
                            Thorne Bay  
                             El Dorado  
                             Pell City  
                          East Brewton  
                          South Tucson  
                              St. Paul  
                    Naberezhnye Chelny  
  Ardatov (town), Republic of Mordovia  
       Babayevo (town), Vologda Oblast  
                      Dabrowa Gornicza  
                            El Bayyada  

* Was person name but predicted as False

Domitian  
                 Caracalla  
                 Aristotle  
                    Seneca  
                   Martial  
                  T-killah  
                Theopompus  
           Gilles de Paris  
              Pinturicchio  
                    Horace  
                     Heino  
  Emperor Yingzong of Song  
 Emperor Guangzong of Song  
    Emperor Duzong of Song  
                     Tolui  
       Alexander the Great  
    Emperor Wuzong of Tang  
                 Lee Hills  
               Tutankhamun  
              Horace Barks  



* Was not person name and predicted as False 

Alexandria  
      Szczecin  
    Versailles  
     Amsterdam  
       Supetar  
        Abakan  
    Appingedam  
     Karlsruhe  
      Dortmund  
        Arnhem  
        MIERDA  
          Juba  
 Charlottetown  
   Fredericton  
     Porsgrunn  
      Vladimir  
        Kaluga  
        Maykop  
      Lilongwe  
        Maseru  


***Please note that all steps can have a lot of improvement, such as:***

* The datasets used can be aligned more to the problem, currently uses a 50/50 of person names and non-person names.
* More data can be used, here 33% (100,000) rows out of 300,000 rows were used.
* More relevent features could be engineered based of google search, parts of speech, dictionaries and using other models to engineer them. 
* The Markov modelling was my first foray in the technique and did not dive deep enough. The model does very well in detecting gibberish. Feel that it can be a very good tool for such usecases and can be used to make it more relevent.
* The feature selection can be improved a great deal. For, example using a glm to select the most valuable and important fearutes based on p-value.
* The deep learning model can be tuned for better performance using the other 77% of data that was not used.



