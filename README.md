# Fake News Detection model
> A Logistic regression model to detect fake news.

## Prerequisites:
- [Dataset](https://www.kaggle.com/datasets/jruvika/fake-news-detection)
- Numpy
- Pandas
- NLTK
- sklearn

## How it's been done?

1. Get all stopwords from english using nltk library  ```nltk.download('stopwords')```
    > Stopwords like i,me,he,you,its,what,did,because etc
2. Reading the dataset file using pandas
3. Checking and Clearing NULL values using  ```news_dataset = news_dataset.fillna('')```
4. Merging Author name and News title to ease the process
5. Separating data and labels
6. **Stemming** -> process of reducing word to its root word by ```port_stem = PorterStemmer()```
    > actor, actress, acting => act is the root word.
7. Create a function and substitute non alphabets by a space
8. Lowercase all letters
9. Split the stemmed_content
10. Convert every word in stem_word which is not stopword, to a list
11. Join all words into a single content and return
12. **Vectorise** the content by converting text into numbers
13. Splitting datasets to training and test data
14. Train the model by instantiating ```model = LogisticRegression()```
15. Now evaluate the model to finout the accuracy score of both training and test data
16. Making a prediction 
 ```
x_new = x_test[320]
prediction = model.predict(x_new)
print(prediction)


![image](https://github.com/vilasrhegde/fakeNewsDetection/assets/85540091/be490bd9-10ae-4346-882e-d51a2e243d79)


if(prediction==0):
  print("News is Real")
else:
  print("News is Fake")  
```
