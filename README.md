# Riiid! Answer Correctness Prediction
**Kaggle competition**

Ranked 138th (Top 4.1%) among 3395 teams

train_lgbm_bayes.py: train the LightGBM model with auto-tunning hyperparameters by Bayes optimization  
train_SAKT.py: train the SAKT model
main.py: main code to evaluate the test data based on LightGBM model and SAKT model.  

The final inference is blended by both LightGBM and SAKT models:  
0.7*LightGBM + 0.3*SAKT
