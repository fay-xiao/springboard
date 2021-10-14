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
   kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
   kmeans.fit(tripadvisor_data1)
   prediction = kmeans.predict(tripadvisor_data1)
   tripadvisor_data['Cluster'] = prediction
   st.write(tripadvisor_data.head(30))

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
   target_names = [0,1,2]
   pca = PCA(n_components=2) # reduce dimesions of the data using PCA and LDA
   X_r = pca.fit(X).transform(X)
   lda = LinearDiscriminantAnalysis(n_components=2)
   X_r2 = lda.fit(X, y).transform(X)

   plt.figure(figsize=(16,8))
   colors = ['#ffa60a', '#880000', '#0a0054',]
   for color, i, target_name in zip(colors, range(0,4), target_names): # plot clusters
      plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.5,
                label=target_name)
   plt.title('Google dataset clusters using PCA')
   st.pyplot(plt)

   kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
   kmeans.fit(google_data1)
   prediction = kmeans.predict(google_data1)
   google_data['Cluster'] = prediction
   st.write(google_data.head(30))

   with features:
      st.header('You can give recommendation for a given user based data collection from other user. For example, can we provide recommend based user 28 for user 8 for Asia')
      def user_recommendationtripadvisor(firstid, secondid):
         # first user ID
         row_firstuser = tripadvisor_data.loc[tripadvisor_data['User ID']==firstid]
         cluster_firstuser = row_firstuser['Cluster'].item()
         # second user ID
         row_seconduser = tripadvisor_data.loc[tripadvisor_data['User ID']==secondid]
         cluster_seconduser = row_seconduser['Cluster'].item()
         if cluster_firstuser == cluster_seconduser:
            return 'Yes'
         else:
            return 'No'
      def user_recommendationgoogle(firstid, secondid):
         # first user ID
         row_firstuser = google_data.loc[google_data['User ID']==firstid]
         cluster_firstuser = row_firstuser['Cluster'].item()
         # second user ID
         row_seconduser = google_data.loc[google_data['User ID']==secondid]
         cluster_seconduser = row_seconduser['Cluster'].item()   
         if cluster_firstuser == cluster_seconduser:
            return 'Yes'
         else:
            return 'No'
      
      continent=st.selectbox('which continent you are traveling to?', options=['Asia','Europe'], index = 0)
      firstuser=st.text_input('User to be recommendated')
      seconduser=st.text_input('User to be used for recommendation')
      while (not firstuser) and (not seconduser):
         continent=st.selectbox('which continent you are traveling to?', options=['Asia','Europe'])
         firstuser=st.text_input('User to be recommendated')
         seconduser=st.text_input('User to be used for recommendation')
      if continent == 'Asia':
         st.write('Asia')
         st.write(user_recommendationtripadvisor(firstuser, seconduser))
      else:
         st.write('Europe')
         st.write(user_recommendationgoogle(firstuser, seconduser))
