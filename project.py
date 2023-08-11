# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:14:11 2023

@author: torbe
"""

import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from scipy import stats
from scipy.stats import norm
from collections import Counter
from wordcloud import WordCloud

# Also need to download these:
# !python -m spacy download en_core_web_sm
# !python -m spacy download de_core_news_sm
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

#### Load Data ################################################################
# Choose folder path
folder_path = 'D:/Organisation/Uni/Wien/PFF II/Project/Data/'

dataframes = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)

        # Read the CSV file and append its DataFrame to the list
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenate all DataFrames into one
df = pd.concat(dataframes, ignore_index=True)

# Cleanup
del(dataframes, filename, file_path)

###############################################################################

#### Extract salary ###########################################################
##  Possible patterns: 
#   Salary: starting at EUR 40.000,- gross p.a.
#   the minimum wage for this position in accordance with the respective collective agreement is EUR 49.458,64,- gross per year

# Extract salary based on keywords like 'salary' for german and english
pattern = r'(?i)\b(?:gehalt|salary|income|wage|jahresbruttoeinkommen|mindestgehalt|pay).*?(\b\d{1,6}(?:[.,]\d{3})*\b)'
df['salary'] = df['description'].str.extract(pattern, re.I)

###############################################################################

#### Clean the salary #########################################################
# Make salary format consistent
df.salary = df.salary.str.replace('[.,]|EUR','', regex=True)
df.salary = df.salary.str.replace(r'\b\d{1,2}\b|\b\d{7,}\b', 'nan', regex=True)

# Drop every job that does not provide a salary or is not full-time as part-time jobs would mess with the results
df2 = df.dropna(subset=['salary'])
df2 = df2[df2['extensions'].apply(lambda x: 'Full-time' in x)]
df2 = df2[df2.salary != 'nan']

# If salary <5 numbers, multiplied with 14 as it is most likely monthly salary
df2.salary = df2.salary.apply(lambda x: int(x) * 14 if len(x) < 5 else x)

# Salary to int and remove salary = 0
df2.salary = df2.salary.astype(int)
df2 = df2[df2.salary != 0]

# Remove Outliers. It might make sense to include them when haing more data, as also jobs with a very high salary are worth analysing, but with my little data it dominates the results very much
z_scores = np.abs(stats.zscore(df2['salary']))
df2 = df2[z_scores < 3]

###############################################################################

#### Plot salaries ############################################################
# Histogram, to get a overview of how the data is structured
salary_values = df2['salary'].tolist()
plt.figure(figsize=(25, 15))
plt.hist(salary_values, bins=30, density=True, alpha=0.6, color='b', label=f'Salary Distribution (N={len(salary_values)}')

# Include Mean and Median
plt.axvline(np.mean(salary_values), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(salary_values):.2f}')
plt.axvline(np.median(salary_values), color='g', linestyle='dashed', linewidth=2, label=f'Median: {np.median(salary_values):.2f}')

# Plot the normal distribution (bell curve)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, np.mean(salary_values), np.std(salary_values))
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')

# Labels and format
plt.xlabel('Salary', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.title('Salary Distribution', fontsize=20)
plt.xticks(np.arange(20000, 90000, 5000), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
###############################################################################

#### Clean description ########################################################
# Desciption to String and lowercase
df2.description = df2.description.astype(str)
df2.description = df2.description.str.lower()

# Remove all non-letter characters and reduce whitespaces
df2['description'] = df2['description'].apply(lambda x: re.sub(r'[\r\n\u200b]+', ' ', x))
df2['description'] = df2['description'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))
df2['description'] = df2['description'].apply(lambda x: re.sub(r'\s+', ' ', x))

# Remove english and german stopwords
stop_words = stopwords.words('english') + stopwords.words('german')
df2['description'] = df2['description'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

# Use lemmatizer to shrinke words to its base (Got the lemmatizing functions from ChatGPT)
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(treebank_tag):
    tag = {'J': wordnet.ADJ,
           'V': wordnet.VERB,
           'N': wordnet.NOUN,
           'R': wordnet.ADV}
    return tag.get(treebank_tag[0].upper(), wordnet.NOUN)
def lemmatize_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    return ' '.join(lemmatizer.lemmatize(tok, get_wordnet_pos(pos)) for tok, pos in pos_tags)

df2['description'] = df2['description'].apply(lemmatize_sentence)

###############################################################################

#### Plot Word Distributions ##################################################
descriptions = df2['description'].tolist()
all_words = ' '.join(descriptions).split()

# Word Frequency Distribution
word_freq = Counter(all_words)
common_words = word_freq.most_common(50)

words = [word[0] for word in common_words]
counts = [word[1] for word in common_words]

# Plotting the most common words
plt.figure(figsize=(25, 15))
plt.bar(words, counts, color='blue')
plt.xlabel('Words', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('Top 20 Most Common Words', fontsize=20)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Word Cloud
wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(' '.join(all_words))
plt.figure(figsize=(25, 15))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Descriptions', fontsize=20)
plt.show()

###############################################################################

#### Data Preprocessing #######################################################
# Use TFIDF to get importance of the different words
vectorizer = TfidfVectorizer()
# vectorizer = TfidfVectorizer(ngram_range=(2,2))

X = vectorizer.fit_transform(df2['description'])
y = df2['salary']

# Split in Train and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

"""
#### SelectKBest to reduce dimensionality, did not work well
k = 100
selector = SelectKBest(score_func=f_regression, k=k)
X = selector.fit_transform(X, y)

# Store selected words to see them later
selected_mask = selector.get_support()
selected_feature_names = [word for word, boolean in zip(vectorizer.get_feature_names_out(), selected_mask) if boolean]
"""

###############################################################################

#### Linear Regression ########################################################

#### All data #################################################################

# Hyperparams for Linear Regression
params_lr = {
    'positive': [True, False],
    'fit_intercept': [True, False]
}

# Randomized Search for Linear Regression
random_search_lr = RandomizedSearchCV(LinearRegression(), param_distributions=params_lr, n_iter=10, cv=kf, verbose=2, random_state=42, n_jobs=-1)
random_search_lr.fit(X_train, y_train)
best_lr = random_search_lr.best_params_

# Linear Regression with best parameters
lr_best = LinearRegression(**best_lr)
lr_best.fit(X_train, y_train)
predictions_lr = lr_best.predict(X_test)

# Store results to compare later
results = {
    "Model": ["Linear Regression"],
    "R^2": [r2_score(y_test, predictions_lr)],
    "MAE": [mean_absolute_error(y_test, predictions_lr)],
    "MSE": [mean_squared_error(y_test, predictions_lr)]
}

# Store words
lr_words = pd.DataFrame({
    'Word': vectorizer.get_feature_names_out(),
    'Effect': lr_best.coef_
}).sort_values(by='Effect', ascending=False) 

# Plot words as Barplot
N = 25
lr_topbot_words = pd.concat([lr_words.head(N), lr_words.tail(N)])
plt.figure(figsize=(25, 15))
plt.barh(lr_topbot_words['Word'], lr_topbot_words['Effect'], color=['red' if effect < 0 else 'green' for effect in lr_topbot_words['Effect']])
plt.xlabel('Effect', fontsize=16)
plt.ylabel('Word', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title('Top 25 Positive and Negative Effect Words | Linear Regression | All Data', fontsize=20)
plt.gca().invert_yaxis()
plt.show()

# Plot predictions against residuals
lr_residuals = y_test - predictions_lr
plt.figure(figsize=(25, 15))
plt.scatter(predictions_lr, lr_residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values', fontsize=16)
plt.ylabel('Residuals', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Residuals vs. Fitted values', fontsize=20)
plt.show()

# Histogram of residuals
plt.figure(figsize=(25, 15))
plt.hist(lr_residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('Histogram of Residuals', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# QQ Plot of residuals
plt.figure(figsize=(25, 15))
stats.probplot(lr_residuals.values.flatten(), plot=plt)
plt.title('QQ-plot of Residuals', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

## -> Distribution looks ok, but QQ Plot suggests polynomal relationship

###############################################################################

#### Simple Polynomal Regression ##############################################
# Degree = 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)

# Linear-Poly Regression with best parameters
lrp_best = LinearRegression(**best_lr)
lrp_best.fit(X_poly, y_train)
predictions_lrp = lrp_best.predict(poly.transform(X_test))

# Store results
results["Model"].extend(["Polynomal Regression, d=2"])
results["R^2"].extend([r2_score(y_test, predictions_lrp)])
results["MAE"].extend([mean_absolute_error(y_test, predictions_lrp)])
results["MSE"].extend([mean_squared_error(y_test, predictions_lrp)])

# Polynomal Regression performs way better than linear, problem is, that it is not that easily interpretable
# It probably makes sense to check for higher degrees as well, but my laptop is not strong enough to do so

###############################################################################

#### Linear Regression with reduced features ##################################

# Use RFE to decrease number of features
rfe = RFE(LinearRegression(), n_features_to_select=50)
X_rfe = rfe.fit_transform(X_train, y_train)
X_rfe_test = rfe.transform(X_test)

# Store words to check them later if needed
selected_features_mask = rfe.support_
selected_feature_names = [feature for feature, selected in zip(vectorizer.get_feature_names_out(), selected_features_mask) if selected]

# Randomized Search for Linear Regression
random_search_lr_rfe = RandomizedSearchCV(LinearRegression(), param_distributions=params_lr, n_iter=10, cv=kf, verbose=2, random_state=42, n_jobs=-1)
random_search_lr_rfe.fit(X_rfe, y_train)
best_lr_rfe = random_search_lr_rfe.best_params_

# Linear Regression with best parameters
lr_rfe_best = LinearRegression(**best_lr_rfe)
lr_rfe_best.fit(X_rfe, y_train)
predictions_lr_rfe = lr_rfe_best.predict(X_rfe_test)

# Store results
results["Model"].extend(["Linear Regression | RFE"])
results["R^2"].extend([r2_score(y_test, predictions_lr_rfe)])
results["MAE"].extend([mean_absolute_error(y_test, predictions_lr_rfe)])
results["MSE"].extend([mean_squared_error(y_test, predictions_lr_rfe)])

# Store words
lr_rfe_words = pd.DataFrame({
    'Word': selected_feature_names,
    'Effect': lr_rfe_best.coef_
}).sort_values(by='Effect', ascending=False) 


# Plot words as Barplot
N = 25
lr_rfe_topbot_words = pd.concat([lr_rfe_words.head(N), lr_rfe_words.tail(N)])
plt.figure(figsize=(25, 15))
plt.barh(lr_rfe_topbot_words['Word'], lr_rfe_topbot_words['Effect'], color=['red' if effect < 0 else 'green' for effect in lr_rfe_topbot_words['Effect']])
plt.xlabel('Effect', fontsize=16)
plt.ylabel('Word', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title('Top 25 Positive and Negative Effect Words | Linear Regression | RFE', fontsize=20)
plt.gca().invert_yaxis()
plt.show()

# Plot predictions against residuals
lr_rfe_residuals = y_test - predictions_lr_rfe
plt.figure(figsize=(25, 15))
plt.scatter(predictions_lr_rfe, lr_rfe_residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values', fontsize=16)
plt.ylabel('Residuals', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Residuals vs. Fitted values', fontsize=20)
plt.show()

# Histogram of residuals
plt.figure(figsize=(25, 15))
plt.hist(lr_rfe_residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('Histogram of Residuals', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# QQ Plot of residuals
plt.figure(figsize=(25, 15))
stats.probplot(lr_rfe_residuals.values.flatten(), plot=plt)
plt.title('QQ-plot of Residuals', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

## Again shows that data is most likely polynomal, also with reduced features

###############################################################################

#### Lasso Regression #########################################################

#### All data #################################################################

# Hyperparams for Lasso Regression
params_lasso = {
    'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    'max_iter': [5000, 10000, 15000],
    'fit_intercept': [True, False],
    'positive': [True, False]
}

# Randomized Search for Lasso Regression
random_search_las = RandomizedSearchCV(Lasso(), param_distributions=params_lasso, n_iter=10, cv=kf, verbose=2, random_state=42, n_jobs=-1)
random_search_las.fit(X_train, y_train)
best_las = random_search_las.best_params_

# Lasso Regression with best parameters
lasso_best = Lasso(**best_las)
lasso_best.fit(X_train, y_train)
predictions_lasso = lasso_best.predict(X_test)

# Store metrices to compare
results["Model"].extend(["Lasso Regression"])
results["R^2"].extend([r2_score(y_test, predictions_lasso)])
results["MAE"].extend([mean_absolute_error(y_test, predictions_lasso)])
results["MSE"].extend([mean_squared_error(y_test, predictions_lasso)])

# Store words
las_words = pd.DataFrame({
    'Word': vectorizer.get_feature_names_out(),
    'Effect': lasso_best.coef_
}).sort_values(by='Effect', ascending=False) 

# Plot words as Barplot
N = 25
las_topbot_words = pd.concat([las_words.head(N), las_words.tail(N)])
plt.figure(figsize=(25, 15))
plt.barh(las_topbot_words['Word'], las_topbot_words['Effect'], color=['red' if effect < 0 else 'green' for effect in las_topbot_words['Effect']])
plt.xlabel('Effect', fontsize=16)
plt.ylabel('Word', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title('Top 25 Positive and Negative Effect Words | Lasso Regression | All Data', fontsize=20)
plt.gca().invert_yaxis()
plt.show()

# Plot predictions against residuals
lasso_residuals = y_test - predictions_lasso
plt.figure(figsize=(25, 15))
plt.scatter(predictions_lasso, lasso_residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values', fontsize=16)
plt.ylabel('Residuals', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Residuals vs. Fitted values', fontsize=20)
plt.show()

# Histogram of residuals
plt.figure(figsize=(25, 15))
plt.hist(lasso_residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('Histogram of Residuals', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# QQ Plot of residuals
plt.figure(figsize=(25, 15))
stats.probplot(lasso_residuals.values.flatten(), plot=plt)
plt.title('QQ-plot of Residuals', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

#### End of 'official' part ###################################################

###############################################################################

###############################################################################

#### Try RFE With Lasso #######################################################
# Careful when running, takes ~1h on my laptop
rfe = RFE(Lasso(max_iter=10000), n_features_to_select=50)
X_las = rfe.fit_transform(X_train, y_train)
X_las_test = rfe.transform(X_test)

selected_features_mask = rfe.support_
selected_feature_names = [feature for feature, selected in zip(vectorizer.get_feature_names_out(), selected_features_mask) if selected]


# Hyperparams for Lasso Regression
params_lasso = {
    'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    'max_iter': [5000, 10000, 15000],
    'fit_intercept': [True, False],
    'positive': [True, False]
}

# Randomized Search for Lasso Regression
random_search_lasso = RandomizedSearchCV(Lasso(), param_distributions=params_lasso, n_iter=20, cv=kf, verbose=2, random_state=42, n_jobs=-1)
random_search_lasso.fit(X_las, y_train)
best_lasso = random_search_lasso.best_params_

# Lasso Regression with best parameters
lasso_best = Lasso(**best_lasso)
lasso_best.fit(X_las, y_train)
predictions_lasso = lasso_best.predict(X_las_test)

results["Model"].extend(["Lasso Regression"])
results["R^2"].extend([r2_score(y_test, predictions_lasso)])
results["MAE"].extend([mean_absolute_error(y_test, predictions_lasso)])
results["MSE"].extend([mean_squared_error(y_test, predictions_lasso)])

# Store words
lasso_words = pd.DataFrame({
    'Word': selected_feature_names,
    'Effect': lasso_best.coef_
}).sort_values(by='Effect', ascending=False)


# Plot words
plt.figure(figsize=(12, 10))
plt.barh(lasso_words['Word'], lasso_words['Effect'], color=['red' if effect < 0 else 'green' for effect in lasso_words['Effect']])
plt.xlabel('Effect')
plt.ylabel('Word')
plt.title('Top 50 Positive and Negative Effect Words')
plt.gca().invert_yaxis()
plt.show()

###############################################################################

#### Ridge Regression #########################################################
rfe = RFE(Ridge(), n_features_to_select=50)
X_rid = rfe.fit_transform(X, y)

selected_features_mask = rfe.support_
selected_feature_names = [feature for feature, selected in zip(vectorizer.get_feature_names_out(), selected_features_mask) if selected]


# Hyperparams for Ridge Regression
params_ridge = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False]
}

# Randomized Search for Ridge Regression
random_search_ridge = RandomizedSearchCV(Ridge(), param_distributions=params_ridge, n_iter=20, cv=kf, verbose=2, random_state=42, n_jobs=-1)
random_search_ridge.fit(X_rid, y)
best_ridge = random_search_ridge.best_params_

# Ridge Regression with best parameters
ridge_best = Ridge(**best_ridge)
ridge_best.fit(X_rid, y)
predictions_ridge = ridge_best.predict(X_rid)

results["Model"].extend(["Ridge Regression"])
results["R^2"].extend([r2_score(y, predictions_ridge)])
results["MAE"].extend([mean_absolute_error(y, predictions_ridge)])
results["MSE"].extend([mean_squared_error(y, predictions_ridge)])

# Store words
ridge_words = pd.DataFrame({
    'Word': selected_feature_names,
    'Effect': ridge_best.coef_
}).sort_values(by='Effect', ascending=False)

# Plot words
plt.figure(figsize=(12, 10))
plt.barh(ridge_words['Word'], ridge_words['Effect'], color=['red' if effect < 0 else 'green' for effect in ridge_words['Effect']])
plt.xlabel('Effect')
plt.ylabel('Word')
plt.title('Top 50 Positive and Negative Effect Words')
plt.gca().invert_yaxis()
plt.show()

###############################################################################

#### Random Forest ############################################################
# Hyperparams for Random Forest Regressor
params_rf = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Randomized Search for Random Forest
random_search_rf = RandomizedSearchCV(RandomForestRegressor(), param_distributions=params_rf, n_iter=100, cv=kf, verbose=2, random_state=42, n_jobs=-1)
random_search_rf.fit(X, y)
best_rf = random_search_rf.best_params_

# Random Forest Regressor with best parameters
rf_best = RandomForestRegressor(**best_rf)
rf_best.fit(X, y)
predictions_rf = rf_best.predict(X)

results["Model"].extend(["Random Forest"])
results["R^2"].extend([r2_score(y, predictions_rf)])
results["MAE"].extend([mean_absolute_error(y, predictions_rf)])
results["MSE"].extend([mean_squared_error(y, predictions_rf)])

# Show words
rf_feature_importances = pd.DataFrame({
    'Word': vectorizer.get_feature_names_out(),
    'Importance': rf_best.feature_importances_
}).sort_values(by='Importance', ascending=False)

# -> Not interpretable as direct effect, therefore not included above

###############################################################################

