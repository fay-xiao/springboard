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
#Asia data set
adataset = st.beta_container()
#Asia model training
amodelTraining = st.beta_container()
#Asia feature
afeatures = st.beta_container()

#Europe data set
edataset = st.beta_container()
#Europe model training
emodelTraining = st.beta_container()
#Europe feature
efeatures = st.beta_container()

footer = st.beta_container()

with header:
   st.title('Travel recommendation for Europe & Asia')
   st.text('This project analyzes 2 user recommendation datasets to determine a most fitted')
   st.text('recommendation for a given user.')
   st.text('Data source: Europe dataset is from Google and Asia dataset is from TripAdvisor.')

with adataset:
   st.header('Asia user dataset analysis')
   st.write('Trip Advisor dataset')
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
   st.write(tripadvisor_data.head())
   st.write('Basic statistic summary of the Asia dataset')
   st.write(tripadvisor_data.describe())
   st.write('Run the Kmeans algorithm with cluster range 2-20 to plot inertias')
   tripadvisor_data1=tripadvisor_data.drop('User ID', axis=1)
   inertias = []
   for i in range(2,20): # run the algo 2-20 to plot inertias 
      model = KMeans(n_clusters = i) 
      model.fit(tripadvisor_data1) # run mode for current k
      inertias.append(model.inertia_)
   plt.figure(figsize=(16,8))
   plt.plot(range(2,20), inertias, 'bx-')
   plt.xlabel('k')
   plt.ylabel('inertias')
   plt.title('The Elbow Method showing the optimal k for the Asia Dataset')
   st.pyplot(plt)
   st.write('Based on the above graph, select optimal cluster = 2 for Trip Advisor (Asia) dataset')
   kmeans = KMeans(n_clusters=2)
   kmeans.fit(tripadvisor_data1)
   clusterNames = kmeans.labels_
   X = tripadvisor_data1
   y = clusterNames
   pca = PCA(n_components=2)
   X_r = pca.fit(X).transform(X)
   n_clusters=2
   plt.figure(figsize=(16,8))
   for i in range(n_clusters):
      plt.scatter(X_r[y == i, 0], X_r[y == i, 1], alpha=.5)
   plt.title('Trip Advisor (Asia) dataset clusters using PCA')
   st.pyplot(plt)

   with amodelTraining:
      st.header('Training with different model and iteration')
      X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.65,test_size=0.35, random_state=42)
      modeltoUse=st.selectbox('which model to use?', options=['LogisticRegression','SGDClassifier'], index = 0)
      numofIteration=st.selectbox('number of iterations?', options=['10000', '5000', '1000', '500','100'], index = 0)
      if (modeltoUse == "SGDClassifier"):
         model_bl = linear_model.SGDClassifier(loss='log', max_iter=int(numofIteration),tol=0.21)
         model_bl.fit(X_train,y_train)
         st.write("Accuracy score:")
         st.write(model_bl.score(X_test,y_test))
      else:
         clf = linear_model.LogisticRegression(multi_class="auto",solver="lbfgs" ,max_iter=int(numofIteration),penalty='l2')
         clf.fit(X_train,y_train)
         st.write("Accuracy score:")
         st.write(clf.score(X_test,y_test))

   with afeatures:
      st.header('Recommendation for Asia user')
      st.write("Using 2 as average rating for Trip Advisor (Asia) data.")
      st.write("Based on the 2 clusters, depend on which cluster user belongs to, find ratio of reviews above 1.")
      people = list(y)
      zero = []
      one = []
      for i in range(len(people)):
         if people[i] == 0:
            zero.append(i)
         elif people[i] == 1:
            one.append(i)

      def recommendation(data):
         # Figure out the ratio of reviews above 1 (2 is a rating of average for Asia data).
         mostLike = []
         for col in tripadvisorColumnLst[1:]:
            mask = (data[col] >= 1)
            try:
               ratio = (mask.value_counts()[True] / (mask.value_counts()[False]+mask.value_counts()[True]))*100
               if ratio > 50:
                  mostLike.append(col)
            except:
               mostLike.append(col)
         return mostLike
      def findrecommendation(index):
         if index in zero:
            return recommendation(tripadvisor_data.iloc[zero])
         elif index in one:
            return recommendation(tripadvisor_data.iloc[one])
         else:
            return none
      selectedUser=st.selectbox('Select the user', tripadvisor_data['User ID'], index = 0, key = 'selected_user')
      st.write(findrecommendation(int(st.session_state.selected_user[5:])))
      st.write("Conclusion: It seems attractions that most people think near average or above in Asia")
      st.write("are dance clubs, resorts, parks, beaches and religious institutions.")

   with edataset:
      st.header('Europe user dataset analysis')
      st.write('Google dataset')
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
      st.write(google_data.head())
      st.write('Basic statistic summary of the Europe dataset')
      st.write(google_data.describe())
      st.write('Run the Kmeans algorithm with cluster range 2-20 to plot inertias')
      google_data1=google_data.drop('User ID', axis=1)
      inertias = []
      for i in range(2,20):
         model = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) 
         model.fit(google_data1) # run mode for current k
         inertias.append(model.inertia_)
      plt.figure(figsize=(16,8))
      plt.plot(range(2,20), inertias, 'bx-')
      plt.xlabel('k')
      plt.ylabel('inertias')
      plt.title('The Elbow Method showing the optimal k for the Europe Dataset')
      st.pyplot(plt)
      st.write('Based on the above graph, select optimal cluster = 3 for Google (Europe) dataset')
      kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
      kmeans.fit(google_data1) # run model
      clusterNames = kmeans.labels_
      X = google_data1
      y = clusterNames
      pca = PCA(n_components=2)
      X_r = pca.fit(X).transform(X)
      lda = LinearDiscriminantAnalysis(n_components=2)
      X_r2 = lda.fit(X, y).transform(X)
      n_clusters=3
      plt.figure(figsize=(16,8))
      for  i in range(n_clusters): # plot clusters
         plt.scatter(X_r[y == i, 0], X_r[y == i, 1], alpha=.5)
      plt.title('Google (Europe) dataset clusters using PCA')
      st.pyplot(plt)

   with emodelTraining:
      st.header('Training with different model and iteration')
      X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.65,test_size=0.35, random_state=42)
      selectedMod=st.selectbox('Select the model', options=['LogisticRegression', 'SGDClassifier'], index = 0)
      selectedIte=st.selectbox('Select iteration', options=['10000', '5000', '1000', '500', '100'], index = 0)
      if (selectedMod == "SGDClassifier"):
         model_bl = linear_model.SGDClassifier(loss='log', max_iter=int(selectedIte),tol=0.21)
         st.write("Accuracy score:")
         model_bl.fit(X_train,y_train)
         st.write(model_bl.score(X_test,y_test))
      else:
         clf = linear_model.LogisticRegression(multi_class="auto",solver="lbfgs" ,max_iter=int(selectedIte),penalty='l2')
         st.write("Accuracy score:")
         clf.fit(X_train,y_train)
         st.write(clf.score(X_test,y_test))

   with efeatures:
      st.header('Recommendation for Europe user')
      st.write("Selecting 3 as average rating for Google (Europe) data.")
      st.write("Based on the 3 clusters, depend on which cluster user belongs to, find ratio of reviews above 1.")
      
      people = list(y)
      zero = []
      one = []
      two = []
      for i in range(len(people)):
         if people[i] == 0:
            zero.append(i)
         elif people[i] == 1:
            one.append(i)
         else:
            two.append(i)
      def recommendation_google(data):
         # Figure out the ratio of reviews above 1 (3 is a rating of average for Europe data).
         mostLike = []
         for col in googleColumnLst[1:]:
            mask = (data[col] >= 2)
            try:
               ratio = (mask.value_counts()[True] / (mask.value_counts()[False]+mask.value_counts()[True]))*100
               if ratio > 50:
                  mostLike.append(col)
            except:
               mostLike.append(col)
         return mostLike
      def findrecommendation_google(index):
         if index in zero:
            return recommendation_google(google_data.iloc[zero])
         elif index in one:
            return recommendation_google(google_data.iloc[one])
         elif index in two:
            return recommendation_google(google_data.iloc[one])
         else:
            return none
      
      userTobe=st.selectbox('Which user', google_data['User ID'], index = 0, key = 'user_tobe')
      st.write(findrecommendation_google(int(st.session_state.user_tobe[5:])))
      st.write("Conclusion: It seems attractions that most people think near average or above in Europe")
      st.write("are parks, theatres, malls, beaches, museums, zoo, restaurants and pubs/bars.")

   with footer:
      st.header('Relationship of the 2 datasets')
      st.write('Travelers in both continents share similar interests in nature such as parks, beaches and in culture such as theatres, museums, religious institutions.')
      st.write('European travelers have high interest in restaurants and pubs/bars than Asia travelers.')
      st.write('Asia travelers have high interest in dance clubs and resorts than European travelers.')
      
     
