import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("UpdatedResumeDataSet.csv")
# print(df.head())

# print(df.shape)  

# print(df['Category'].value_counts())

# plt.figure(figsize=(15,5))
# sns.countplot(df["Category"])
# plt.xticks(rotation=90)
# plt.show()

# counts = df['Category'].value_counts()
# labels = df['Category'].unique()
# plt.figure(figsize=(15,10))
# plt.pie(counts,labels=labels,autopct='%1.1f%%',shadow=True)
# plt.show()

# print(df['Category'][0])

# print(df['Resume'][0])


# function to filter all the extra key words from the resume so the resume seems clean.
import re
def CleanResume(txt):
  cleanTxt = re.sub('http\S+\s',' ',txt)
  cleanTxt = re.sub('RT|cc',' ',cleanTxt)
  cleanTxt = re.sub('#\S+\s',' ',cleanTxt)
  cleanTxt = re.sub('@\S+',' ',cleanTxt)
  cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_'{|}~"""),' ',cleanTxt)
  cleanTxt = re.sub(r'[^\x00-\x7f]',' ',cleanTxt)
  cleanTxt = re.sub('\s+',' ',cleanTxt)
  return cleanTxt

# print(df['Resume'])

df['Resume'] = df['Resume'].apply(lambda x:CleanResume(x))

# print(df['Resume'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])

# print(df)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

tfidf.fit(df['Resume'])
requiredText = tfidf.transform(df['Resume'])

# print(df)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(requiredText, df['Category'], test_size=0.2, random_state=42)

# print(x_train.shape)
# print(x_test.shape)  

# now let's train the model and print the classification report:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer

print("lendi")
# Initialize TfidfVectorizer
tfidfd = TfidfVectorizer()

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(x_train,y_train)
ypred = clf.predict(x_test)
# print(ypred) 
# print(accuracy_score(y_test,ypred))


# now creating the prediction system
# print(tfidf)

import pickle
pickle.dump(tfidf,open('tfidf.pkl','wb'))
pickle.dump(clf,open('clf.pkl','wb'))

myresume = """I am a data scientist specializing in machine
learning, deep learning, and computer vision. With
a strong background in mathematics, statistics,
and programming, I am passionate about
uncovering hidden patterns and insights in data.
I have extensive experience in developing
predictive models, implementing deep learning
algorithms, and designing computer vision
systems. My technical skills include proficiency in
Python, Sklearn, TensorFlow, and PyTorch.
What sets me apart is my ability to effectively
communicate complex concepts to diverse
audiences. I excel in translating technical insights
into actionable recommendations that drive
informed decision-making.
If you're looking for a dedicated and versatile data
scientist to collaborate on impactful projects, I am
eager to contribute my expertise. Let's harness the
power of data together to unlock new possibilities
and shape a better future.
Contact & Sources
Email: 611noorsaeed@gmail.com
Phone: 03442826192
Github: https://github.com/611noorsaeed
Linkdin: https://www.linkedin.com/in/noor-saeed654a23263/
Blogs: https://medium.com/@611noorsaeed
Youtube: Artificial Intelligence
ABOUT ME
WORK EXPERIENCE
SKILLES
NOOR SAEED
LANGUAGES
English
Urdu
Hindi
I am a versatile data scientist with expertise in a wide
range of projects, including machine learning,
recommendation systems, deep learning, and computer
vision. Throughout my career, I have successfully
developed and deployed various machine learning models
to solve complex problems and drive data-driven
decision-making
Machine Learnine
Deep Learning
Computer Vision
Recommendation Systems
Data Visualization
Programming Languages (Python, SQL)
Data Preprocessing and Feature Engineering
Model Evaluation and Deployment
Statistical Analysis
Communication and Collaboration
"""


import pickle

# load the trained classifier
clf = pickle.load(open('clf.pkl','rb'))

# clean the input resume
cleaned_resume = CleanResume(myresume)
# print(cleaned_resume)

# transform the cleaned resume using the trained TfidfVectorizer
input_features = tfidf.transform([cleaned_resume])

# print(input_features)

prediction_id = clf.predict(input_features)[0]
# print(prediction_id)

# mapping category id to category name

category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

category_name = category_mapping.get(prediction_id, "Unknown")
print("Predicted Category :", category_name)
