#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:04:12 2019

@author: sanjitadvani
"""

#Importing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

#Importing dataset
survey = pd.read_excel('finalExam_Mobile_App_Survey_Data.xlsx')

###########################################################################
#exploratory
survey.info()
survey_describe=survey.describe()

#Creating histograms for each variable
survey.hist(bins=50, figsize=(20,20))
plt.show()

# Check for NANs values
nan = survey.isna().sum()
nan[nan > 0]

# Check for Missing values 
print(
      survey.columns
      .isnull()
      .sum()
      )

############################################################################
#PCA analysis
#Renaming demographics columns
survey.rename(columns={'q1':'age',
                       'q48':'education',
                       'q49':'marital',
                       'q54':'race',
                       'q55':'ethnicity',
                       'q56':'income',
                       'q57':'gender',
                       "q50r1" : "No Kids",
                       "q50r2" : "Under 6",
                       "q50r3" : "6-12 years old",
                       "q50r4" : "12-18 years old",
                       "q50r5" : "18 and older"},inplace=True)

#Remove demographic information and noisy columns
survey_reduced = survey.drop(columns=['age',
                                      'education',
                                      'marital',
                                      'race',
                                      'ethnicity',
                                      'income',
                                      'gender',
                                      'caseID',
                                      "No Kids",
                                      "Under 6",
                                      "6-12 years old",
                                      "12-18 years old",
                                      "18 and older"])

#Scale to get equal variance
scaler = StandardScaler()
scaler.fit(survey_reduced)
X_scaled_reduced = scaler.transform(survey_reduced)


#Run PCA without limiting the number of components
survey_pca_reduced = PCA(n_components = None,
                           random_state = 508)


survey_pca_reduced.fit(X_scaled_reduced)


X_pca_reduced = survey_pca_reduced.transform(X_scaled_reduced)

#Analyze the scree plot to determine how many components to retain
fig, ax = plt.subplots(figsize=(10, 8))
features = range(survey_pca_reduced.n_components_)

plt.plot(features,
         survey_pca_reduced.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')

plt.title('survey scree plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()

#look at components when they total to 0.8

#Run PCA again based on the desired number of components
survey_pca_reduced = PCA(n_components = 5,
                           random_state = 508)

survey_pca_reduced.fit(X_scaled_reduced)

#Analyze factor loadings to understand principal components
factor_loadings_df = pd.DataFrame(pd.np
                                  .transpose(survey_pca_reduced.components_))

factor_loadings_df = factor_loadings_df.set_index(survey_reduced.columns[0:])

print(factor_loadings_df)

factor_loadings_df.to_excel('factor_loadings.xlsx')

#Analyze factor strengths per customer
X_pca_reduced = survey_pca_reduced.transform(X_scaled_reduced)
X_pca_df = pd.DataFrame(X_pca_reduced)

#Rename your principal components and reattach demographic information
X_pca_df.columns = ['comp1', 'comp2', 'comp3','comp4','comp5']

#adding the demographic columns again
final_pca_df = pd.concat([survey.loc[ : , 
                        ['age', 'income','gender','education']] , X_pca_df], axis = 1)

# Analyze in more detail
# Renaming channels
age_range = {1 : 'under 18',
             2 : '18-24',
             3 : '25-29',
             4 : '30-34',
             5 : '35-39',
             6 : '40-44',
             7 : '45-49',
             8 : '50-54',
             9 : '55-59',
             10 : '60-64',
             11 : '65 and older'}
final_pca_df['age'].replace(age_range, inplace = True)

# Renaming Education
education_range = {1 : 'some high school',
                   2 : 'high school graduate',
                   3 : 'some college',
                   4 : 'college graduate',
                   5 : 'some post-graduate',
                   6 : 'post graduate degree'}
final_pca_df['education'].replace(education_range, inplace = True)

# Renaming Income
income_range = {1 : 'under $10,000',
                2 : '10,000-14,999',
                3 : '15,000-19,999',
                4 : '20,000-29,999',
                5 : '30,000-39,999',
                6 : '40,000-49,999',
                7 : '50,000-59,999',
                8 : '60,000-69,999',
                9 : '70,000-79,999',
                10 : '80,000-89,999',
                11 : '90,000-99,999',
                12 : '100,000-124,999',
                13 : '125,000-149,999',
                14 : '150,000-Over'}
final_pca_df['income'].replace(income_range, inplace = True)

# Renaming Gender
gender = {1 : 'Male',
          2 : 'Female'}
final_pca_df['gender'].replace(gender, inplace = True)

# Analyzing by age
fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(x = 'age',
            y =  'comp1',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(x = 'age',
            y =  'comp2',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(x = 'age',
            y =  'comp3',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(x = 'age',
            y =  'comp4',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(x = 'age',
            y =  'comp5',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

# Analyzing by income
fig, ax = plt.subplots(figsize = (15, 5))
sns.boxplot(x = 'income',
            y =  'comp1',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (15, 5))
sns.boxplot(x = 'income',
            y =  'comp2',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (15, 5))
sns.boxplot(x = 'income',
            y =  'comp3',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (15, 5))
sns.boxplot(x = 'income',
            y =  'comp4',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize = (15, 5))
sns.boxplot(x = 'income',
            y =  'comp5',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

# Analyzing by gender
fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(x = 'gender',
            y =  'comp1',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(x = 'gender',
            y =  'comp2',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(x = 'gender',
            y =  'comp3',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(x = 'gender',
            y =  'comp4',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(x = 'gender',
            y =  'comp5',
            data = final_pca_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

##K-means clustering

#Experiment with different numbers of clusters
survey_k = KMeans(n_clusters = 5,
                      random_state = 508)
survey_k.fit(X_scaled_reduced)
survey_kmeans_clusters = pd.DataFrame({'cluster': survey_k.labels_})
print(survey_kmeans_clusters.iloc[: , 0].value_counts())

# Step 4: Analyze cluster centers
centroids = survey_k.cluster_centers_
centroids_df = pd.DataFrame(centroids)

# Renaming columns
centroids_df.columns = survey_reduced.columns
print(centroids_df)

# Sending data to Excel
centroids_df.to_excel('survey_k3_centriods.xlsx')

#Analyze cluster memberships
X_scaled_reduced_df = pd.DataFrame(X_scaled_reduced)
X_scaled_reduced_df.columns = survey_reduced.columns
clusters_df = pd.concat([survey_kmeans_clusters,
                         X_scaled_reduced_df],
                         axis = 1)
print(clusters_df)

#Reattach demographic information 
final_clusters_df = pd.concat([survey.loc[ : , ['age', 'income','education',
                                                'gender'] ],
                               clusters_df],
                               axis = 1)
print(final_clusters_df)

# Analyze in more detail 
# Renaming age
age_range = {1 : 'Under 18',
             2 : '18-24',
             3 : '25-29',
             4 : '30-34',
             5 : '35-39',
             6 : '40-44',
             7 : '45-49',
             8 : '50-54',
             9 : '55-59',
             10 : '60-64',
             11 : '65 and older'}

final_clusters_df['age'].replace(age_range, inplace = True)

# Renaming education
education_range = {1 : 'Some High School',
                   2 : 'High School Graduate',
                   3 : 'Some College',
                   4 : 'College Graduate',
                   5 : 'Some Post-graduate',
                   6 : 'Post Graduate Degree'}

final_clusters_df['education'].replace(education_range, inplace = True)

# Renaming Income
income_range = {1 : 'under $10,000',
                2 : '10,000-14,999',
                3 : '15,000-19,999',
                4 : '20,000-29,999',
                5 : '30,000-39,999',
                6 : '40,000-49,999',
                7 : '50,000-59,999',
                8 : '60,000-69,999',
                9 : '70,000-79,999',
                10 : '80,000-89,999',
                11 : '90,000-99,999',
                12 : '100,000-124,999',
                13 : '125,000-149,999',
                14 : '150,000-Over'}
final_pca_df['income'].replace(income_range, inplace = True)

# Renaming Gender
gender = {1 : 'Male',
          2 : 'Female'}
final_pca_df['gender'].replace(gender, inplace = True)

#analysis by Device - iPhone
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y = 'q2r1',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# Device 2 - iPod touch
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y = 'q2r2',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 6)
plt.tight_layout()
plt.show()

# Device 3 - Androic
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y = 'q2r3',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 6)
plt.tight_layout()
plt.show()

# Device 4 - BB
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y = 'q2r4',
            hue = 'cluster',
            data = final_clusters_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# Nokia
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y = 'q2r5',
            data = final_clusters_df)

plt.ylim(-2, 5)
plt.tight_layout()
plt.show()

# Windows
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y = 'q2r6',
            data = final_clusters_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

#combining PCA and K-means clustering
# Take your transformed dataframe
print(X_pca_df.head(n = 5))
print(pd.np.var(X_pca_df))

# Scale to get equal variance
scaler = StandardScaler()
scaler.fit(X_pca_df)
X_pca_clust = scaler.transform(X_pca_df)
X_pca_clust_df = pd.DataFrame(X_pca_clust)
print(pd.np.var(X_pca_clust_df))
X_pca_clust_df.columns = X_pca_df.columns


#Experiment with different numbers of clusters
survey_k_pca = KMeans(n_clusters = 5,
                         random_state = 508)
survey_k_pca.fit(X_pca_clust_df)

survey_kmeans_pca = pd.DataFrame({'cluster': survey_k_pca.labels_})

print(survey_kmeans_pca.iloc[: , 0].value_counts())

# Analyze cluster centers
centroids_pca = survey_k_pca.cluster_centers_
centroids_pca_df = pd.DataFrame(centroids_pca)

# Rename your principal components
centroids_pca_df.columns = ['comp1', 'comp2', 'comp3','comp4','comp5']
print(centroids_pca_df)

# Sending data to Excel
centroids_pca_df.to_excel('survey_pca_centriods.xlsx')

#Analyze cluster memberships
clst_pca_df = pd.concat([survey_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)
print(clst_pca_df)

# Reattach demographic information
final_pca_clust_df = pd.concat([survey.loc[ : , ['age', 'income','education']],
                                clst_pca_df],
                                axis = 1)
print(final_pca_clust_df.head(n = 5))

#Analyze in more detail 
# Renaming Age
age_range = {1 : 'Under 18',
             2 : '18-24',
             3 : '25-29',
             4 : '30-34',
             5 : '35-39',
             6 : '40-44',
             7 : '45-49',
             8 : '50-54',
             9 : '55-59',
             10 : '60-64',
             11 : '65 and older'}

final_pca_clust_df['age'].replace(age_range, inplace = True)

# Renaming Education
education_range = {1 : 'Some High School',
                   2 : 'High School Graduate',
                   3 : 'Some College',
                   4 : 'College Graduate',
                   5 : 'Some Post-graduate',
                   6 : 'Post Graduate Degree'}

final_pca_clust_df['education'].replace(education_range, inplace = True)

# Renaming income
income_range = {1 : 'Under $10,000',
                2 : '10,000-14,999',
                3 : '15,000-19,999',
                4 : '20,000-29,999',
                5 : '30,000-39,999',
                6 : '40,000-49,999',
                7 : '50,000-59,999',
                8 : '60,000-69,999',
                9 : '70,000-79,999',
                10 : '80,000-89,999',
                11 : '90,000-99,999',
                12 : '100,000-124,999',
                13 : '125,000-149,999',
                14 : '150,000-Over'}

final_pca_clust_df['income'].replace(income_range, inplace = True)

# Adding a productivity step
data_df = final_pca_clust_df
data_df.head()

data_df=data_df.rename(columns = {0 :'comp1',
                          1 : 'comp2',
                          2 : 'comp3',
                          3 : 'comp4',
                          4 : 'comp5'})

# analysing by age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y = 'comp1',
            hue = 'cluster',
            data = data_df)

plt.ylim(-2, 9)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y = 'comp2',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 6)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y = 'comp3',
            hue = 'cluster',
            data = data_df)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y = 'comp4',
            hue = 'cluster',
            data = data_df)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y = 'comp5',
            hue = 'cluster',
            data = data_df)

plt.ylim(-5, 3)
plt.tight_layout()
plt.show()

# income
fig, ax = plt.subplots(figsize = (15, 5))
sns.boxplot(x = 'income',
            y = 'comp1',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 4)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'income',
            y = 'comp2',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 5)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'income',
            y = 'comp3',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 3)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'income',
            y = 'comp4',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 3)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'income',
            y = 'comp5',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 3)
plt.tight_layout()
plt.show()

############################################################################
