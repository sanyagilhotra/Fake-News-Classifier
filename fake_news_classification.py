# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# %%
true_news=pd.read_csv("True.csv")
fake_news=pd.read_csv("Fake.csv")

# %%
true_news.head()

# %%


# Assuming true_news is your DataFrame
subject_counts = true_news['subject'].value_counts()

# Plotting the bar graph
# plt.figure(figsize=(10, 6))
# subject_counts.plot(kind='bar', color='lightgreen')
# plt.title('Distribution of Subjects in True News')
# plt.xlabel('Subject')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()

# %%
# fake_news.head()

# %%
# subject_counts = fake_news['subject'].value_counts()

# # Plotting the bar graph
# plt.figure(figsize=(10, 6))
# subject_counts.plot(kind='bar', color='pink')
# plt.title('Distribution of Subjects in True News')
# plt.xlabel('Subject')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()

# %%
fake_news['class']=0
true_news['class']=1

# %%
# fake_news.shape

# # %%
# true_news.shape

# %%
news_df=pd.concat([true_news, fake_news], axis = 0)
columns=news_df.columns

# %%
# print(columns)

# %%
news_df.isnull().sum()

# %%
news_df=news_df.drop(['date','subject'],axis=1)
news_df=news_df.sample(frac=1)

# %%
news_df.head()

# %%
news_df.reset_index(inplace = True)

# %%
news_df.head()

# %%
news_df=news_df.drop('index',axis=1)

# %%
random_values = news_df.sample(n=10)
news_df=news_df.drop(random_values.index)

# %%
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove square brackets and their content
    text = re.sub('\[.*?\]', '', text)
    
    # Remove non-word characters (except spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove newlines
    text = text.replace('\n', '')
    
    # Remove words containing digits
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    
    return text


# %%
def remove_stopwords(text):
    word=text.split()
    stop_words = set(stopwords.words('english'))
    filtered_text = ""
    for w in word:
        if w not in stop_words:
            filtered_text=filtered_text+" "+w;
    return filtered_text

# %%
def lemmatised_words(text):
    word=text.split()
    lemmatizer = WordNetLemmatizer()
    lemmatized_sen=""
    for w in word:
        lemmatized_sen= lemmatized_sen+" "+lemmatizer.lemmatize(w)
    return lemmatized_sen

# %%
news_df['text']=news_df['text'].apply(clean_text)

# %%
news_df.head()

# %%
news_df['text']=news_df['text'].apply(remove_stopwords)

# %%
news_df.head()

# %%
news_df['text']=news_df['text'].apply(lemmatised_words)

# %%
news_df.head()

# %%
X = news_df['text'].values
Y = news_df['class'].values

# %%
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# %%
vectorization = TfidfVectorizer()
XV_train = vectorization.fit_transform(X_train)
XV_test = vectorization.transform(X_test)

# %%
dt=DecisionTreeClassifier(random_state=0)
dt.fit(XV_train,Y_train)


# %%
y_pred=dt.predict(XV_test)
print(y_pred)

# %%
cm=confusion_matrix(Y_test,y_pred)
plt.title("Confusion Matrix")
sns.heatmap(pd.DataFrame(cm), annot=True,lw=2)

# %%
def classify_news(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(clean_text)
    new_def_test["text"] = new_def_test["text"].apply(remove_stopwords)
    new_def_test["text"] = new_def_test["text"].apply(lemmatised_words)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_DT = dt.predict(new_xv_test)
    return pred_DT
    


