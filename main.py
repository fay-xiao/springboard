import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import random
from sklearn import linear_model, metrics, model_selection
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors

header =  st.beta_container()
dataset = st.beta_container()
features = st.beta_container()

with header:
   st.title('Travel recommendation for Europe & Asia')
   st.text('This project uses 2 user recommendation datasets to determine a most fitted recommendation for a given user. Europe dataset is from Google and Asia dataset is from TripAdvisor.')

with dataset:
   st.header('Asia user recommendation dataset')
   tripadvisor_file = "tripadvisor_review.csv"
   tripadvisor_data = pd.read_csv(tripadvisor_file)
   tripadvisorColumnLst=[
    'User ID',
    'art galleries',
    'dance clubs',
    'juice bars',
    'restaurants',
    'museums',
    'resorts',
    'parks',
    'beaches',
    'theatres',
    'religious institutions',
   ]
   tripadvisor_data.columns = tripadvisorColumnLst
   st.write(tripadvisor_data.head(20))
   st.write('Basic statistic summary of the dataset')
   st.write(tripadvisor_data.describe())
   st.write('plotting with Silhouette Score with cluster value from 2 to 19')
   tripadvisor_data1=tripadvisor_data.drop('User ID', axis=1)
   s_score = [] # create empty list
   for i in range(2, 20): # for each value from 2 to 19:
      kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
      kmeans.fit(tripadvisor_data1)
      pred=kmeans.predict(tripadvisor_data1)
      s_score.append(metrics.silhouette_score(tripadvisor_data1, pred))
   plt.plot(range(2, 20), s_score, marker='o', c='coral')
   plt.title('The Silhouette Score')
   plt.xlabel('Number of Clusters')
   plt.ylabel('Silhouette Score')
   st.pyplot(plt)
   st.write('Set k = 3')
   kmeans = KMeans(n_clusters=3, init='k-means++', random_state=101)
   kmeans.fit(tripadvisor_data1)
   prediction = kmeans.predict(tripadvisor_data1)
   X = tripadvisor_data1
   y = prediction
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.65,test_size=0.35, random_state=101)
   pca = PCA(n_components=2) # reduce dimesions of the data using PCA and LDA
   X_r = pca.fit(X).transform(X)
   lda = LinearDiscriminantAnalysis(n_components=2)
   X_r2 = lda.fit(X, y).transform(X)
   n_clusters=3
   plt.figure(figsize=(16,8))
   for  i in range(n_clusters): # plot clusters
      plt.scatter(X_r[y == i, 0], X_r[y == i, 1], alpha=.5)

   plt.title('Trip Advisor dataset clusters using PCA')
   st.pyplot(plt)
   

   st.header('Europe user recommendation dataset')
   google_file = "google_review_ratings.csv"
   google_data = pd.read_csv(google_file)
   google_data=google_data.drop(["Unnamed: 25"],axis=1)
   googleColumnLst=[
    'User ID',
    'churches',
    'resorts',
    'beaches',
    'parks',
    'theatres',
    'museums',
    'malls',
    'zoo',
    'restaurants',
    'pubs/bars',
    'local services',
    'burger/pizza shops',
    'hotels/other lodgings',
    'juice bars',
    'art galleries',
    'dance clubs',
    'swimming pools',
    'gyms',
    'bakeries',
    'beauty & spas',
    'cafes',
    'view points',
    'monuments',
    'gardens',
   ]
   google_data.columns = googleColumnLst
   google_data.fillna(0,inplace=True)
   local_services_mean = google_data['local services'][google_data['local services'] != '2\t2.']
   google_data['local services'][google_data['local services'] == '2\t2.'] = np.mean(local_services_mean.astype('float64'))
   google_data['local services'] = google_data['local services'].astype('float64')
   pd.set_option('display.max_columns', 30)
   st.write(google_data.head(20))
   st.write('Basic statistic summary of the dataset')
   st.write(google_data.describe())
   st.write('plotting with inertias with cluster value from 2 to 19')
   google_data1=google_data.drop('User ID', axis=1)
   inertias = []
   for i in range(2,20): # run the algo 2-20 to plot inertias 
      model = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) 
      model.fit(google_data1) # run mode for current k
      inertias.append(model.inertia_)
   plt.figure(figsize=(16,8))
   plt.plot(range(2,20), inertias, 'bx-')
   plt.xlabel('k')
   plt.ylabel('inertias')
   plt.title('The Elbow Method showing the optimal k for the Google Dataset')
   st.pyplot(plt)
   st.write('Set k = 3')
   kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42) # set clusters = 3
   kmeans.fit(google_data1) # run model
   clusterNames = kmeans.labels_
   X = google_data1
   y = clusterNames
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.65,test_size=0.35, random_state=42)
   pca = PCA(n_components=2) # reduce dimesions of the data using PCA and LDA
   X_r = pca.fit(X).transform(X)
   lda = LinearDiscriminantAnalysis(n_components=2)
   X_r2 = lda.fit(X, y).transform(X)
   n_clusters=3
   plt.figure(figsize=(16,8))
   for  i in range(n_clusters): # plot clusters
      plt.scatter(X_r[y == i, 0], X_r[y == i, 1], alpha=.5)
   plt.title('Google dataset clusters using PCA')
   st.pyplot(plt)

   with features:
      st.header('Recommendation for a given user')
      input_data_matrix = google_data[googleColumnLst[1:]].values
      knn_model = NearestNeighbors(n_neighbors=3).fit(input_data_matrix)
      def compare_dfgoogle(index, ind):   
          zero_cols_in = google_data.loc[index].astype(bool)
          zero_df_in = pd.DataFrame(zero_cols_in[zero_cols_in == True]).reset_index(level = 0)
          in_wo_rating = zero_df_in['index']
          sug_user = google_data.loc[ind]
          zero_cols_sug = sug_user.astype(bool)
          zero_df_sug = pd.DataFrame(zero_cols_sug[zero_cols_sug == True]).reset_index(level = 0)
          sug_wo_rating = zero_df_sug['index']
          sugg_list = list(set(sug_wo_rating) - set(in_wo_rating))
          return sugg_list
      def recommend_knngoogle(index):
          distances, indices = knn_model.kneighbors(google_data[googleColumnLst[1:]].iloc[index, :].values.reshape(1,-1), n_neighbors = 10)
          distances = np.sort(distances)
          for i in range(0,len(indices[0])):
             ind = np.where(distances.flatten() == distances[0][i])[0][0]
             sug_list = compare_dfgoogle(index, indices[0][i]) 
             if len(sug_list) > 0:
                 break
          return sug_list
      
      continent=st.selectbox('which continent you are traveling to?', options=['Asia','Europe'], index = 0)
      selectedUser=st.text_input('Enter the user')
      st.write(recommend_knngoogle(int(selectedUser)))
      
     
