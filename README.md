# springboard
##Capstone project: Travel Recommendation with multi-datasets

##Datasets used:
* tripadvisor_review.csv is from https://www.kaggle.com/prashanthv945/travel-reviews is a collection of ratings on different categories from Asia travelers. Rated categories are: art galleries, dance clubs, juice bars, restaurants,museums,resorts,parks,
beaches,theatres and religious institutions. Rating is in a range from 0 - 4 and we use 2 as the average rating.
* google_review_ratings.csv is from https://www.kaggle.com/wirachleelakiatiwong/travel-review-rating-dataset is a collection of ratings on different categories from Europe travelers. Rated categories are: churches,resorts,beaches,parks,theatres,museums,malls,zoo,restaurants,pubs/bars,local services,burger/pizza shops,hotels/other lodgings,juice bars,art galleries,dance clubs,swimming pools,gyms,bakeries,beauty & spas,cafes,view points,monuments and gardens. Rating is in a range from 0 - 5 and we use 3 as the average rating.

##Important files:
* jupyter notebook: recommendationwithbothgoogleNtripadvisorusingkmeancluster.ipynb
* streamlit: main.py
* requirements.txt used to specify needed python libraries for streamlit to pick up

##Process:
* 1. we clean up the datesets and take a look the basic statistics of each set
* 2. We find inertias with different cluster range 2 - 20 to find the optimal k. Displayed in a line graph.
* 3. We select the optimal k. for Tripadvisor (Asia) set, we choose 2 and for Google (Europe) set, we choose 3.
* 4. Using k means clustering (k=2 for Tripadvisor & k=3 form Google), we display the dataset in cluster graph.
* 5. We train the data with SGDClassifier, LogisticRegression with different iteration and displays its accuracy.
* 6. The feature is to select a user from the dataset and displays recommended categories.
* 7. Conclusion on individual dataset: 
        Attractions that most people think near average or above in Europe are parks, theatres, malls, beaches, museums, zoo, restaurants and pubs/bars.
        Attractions that most people think near average or above in Asia are dance clubs, resorts, parks, beaches and religious institutions.
* 8. Conclusion on relationship of both dataset: 
        Travelers in both continents share similar interests in nature such as parks, beaches and in culture such as theatres, museums, religious institutions.
        European travelers have high interest in restaurants and pubs/bars than Asia travelers.
        Asia travelers have high interest in dance clubs and resorts than European travelers.

##Validation:
* Run main.py on streamlit. select different training model and different iteration to see the result of accuracy.
* Run main.py on streamlit. select different user for each dataset and see recommended categories being displayed.
* Insert few new records into the dataset, see if correct recommended categories is displayed.


