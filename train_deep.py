#IMPORT ALL THE THINGS
import h2o
import numpy as np
import pandas as pd
import os

from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator



h2o.init(max_mem_size = 4)
h2o.remove_all()

df = pd.read_csv('./datasets/pos_dictionary_validity.csv')
df['last_name_low_tag'] = df['last_name_low_tag'] == 0
df['is_person'] = df['is_person'] == 1


feature_df = h2o.H2OFrame(df)
#print(feature_df.head())
del df



train, valid, test = feature_df.split_frame([0.6, 0.2], seed=1234)
feature_df_X = feature_df.col_names[3:26] 
feature_df_Y = feature_df.col_names[1]  

#print(feature_df_X)
#print(feature_df_Y)

dl_v1 = H2ODeepLearningEstimator(model_id="dl_v1", epochs=1, variable_importances=True)
dl_v1.train(feature_df_X, feature_df_Y, training_frame = train, validation_frame = valid)
result_1 = pd.DataFrame(dl_v1.varimp(), columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])
#print(result_1)




dl_v1 = H2ODeepLearningEstimator(
    model_id="dl_v1", 
    hidden=[32,32,32],                  ## small network, runs faster
    epochs=1000,                      ## hopefully converges earlier...
    score_validation_samples=1000,      ## sample the validation dataset (faster)
    stopping_rounds=2,
    stopping_metric="misclassification",    
    stopping_tolerance=0.01)
dl_v1.train(feature_df_X, feature_df_Y, training_frame=train, validation_frame=valid)
#print(dl_v1.score_history())
#print(dl_v1)


#print(test[3:26])
pred = dl_v1.predict(test[3:26]).as_data_frame(use_pandas=True)
test_actual = test.as_data_frame(use_pandas=True)['is_person']
#print(pred.head())
#test_actual['predicted'] = pred['predict']
test_actual = pd.concat([test_actual.reset_index(drop=True), pred], axis=1)

people =len(test_actual[test_actual['is_person'] == True])
people_true = len(test_actual.query('is_person == True & predict == True'))
print("Was person name and predicted as True   "+str((people_true/people)*100))
nonpeople =len(test_actual[test_actual['is_person'] == False])
non_people_true = len(test_actual.query('is_person == False & predict == True'))
print("Was not person name but predicted as True   "+str((non_people_true/nonpeople)*100))

people =len(test_actual[test_actual['is_person'] == True])
people_false = len(test_actual.query('is_person == True & predict == False'))
print("Was person name but predicted as False   "+str((people_false/people)*100))
nonpeople =len(test_actual[test_actual['is_person'] == False])
non_people_false = len(test_actual.query('is_person == False & predict == False'))
print("Was not person name and predicted as False   "+str((non_people_false/nonpeople)*100))

test_actual['str_of_words'] = test.as_data_frame(use_pandas=True)['str_of_words']
print(test_actual.query('is_person == True & predict == True').head(20))
print(test_actual.query('is_person == False & predict == True').head(20))
print(test_actual.query('is_person == True & predict == False').head(20))
print(test_actual.query('is_person == False & predict == False').head(20))

# save the model
model_path = h2o.save_model(model=dl_v1, path="./models", force=True)
#print(dl_v1)
print( model_path)
#./models/dl_v1

      
h2o.cluster().shutdown(prompt=True)
#h2o.shutdown(prompt=True)

