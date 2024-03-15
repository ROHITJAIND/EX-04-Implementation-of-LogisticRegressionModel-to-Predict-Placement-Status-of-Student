# EX-04 Implementation of Logistic Regression Model to Predict the Placement Status of Student
### Ain:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
### Algorithm
1. Get the data and use label encoder to change all the values to numeric.
2. Drop the unwanted values,Check for NULL values, Duplicate values.
3. Classify the training data and the test data.
4. Calculate the accuracy score, confusion matrix and classification report.
### Program:
```Python
import pandas as pd
df=pd.read_csv('CSVs/Placement_Data.csv')
df.head()
df=df.drop(['sl_no','salary'],axis=1)
df.isnull().sum()
df.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
l=["gender","ssc_b","hsc_b","hsc_s","degree_t","workex","specialisation","status"]
for i in l:
    df[i]=le.fit_transform(df[i])
df.head()
x=df.iloc[:,:-1]
x.head()
y=df["status"]
y.head()
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear") 
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Confusion matrix:\n",confusion)
from sklearn.metrics import classification_report
ClsfctonReprt=classification_report(y_test,y_pred)
print(ClsfctonReprt)
```
### Output:
<img width=78% valign="top" src="https://github.com/ROHITJAIND/EX-04-Implementation-of-LogisticRegressionModel-to-Predict-Placement-Status-of-Student/assets/118707073/95efcb85-de2d-4d2e-aa2f-fcd7c877ccec">

<img width=18% src="https://github.com/ROHITJAIND/EX-04-Implementation-of-LogisticRegressionModel-to-Predict-Placement-Status-of-Student/assets/118707073/2804d344-d07e-4072-b077-15d0bf4abb66">

<img height=10% width=48% src="https://github.com/ROHITJAIND/EX-04-Implementation-of-LogisticRegressionModel-to-Predict-Placement-Status-of-Student/assets/118707073/f978df0f-2acd-485a-a877-0ff84d2f9b8e">

<img height=10% width=48% src="https://github.com/ROHITJAIND/EX-04-Implementation-of-LogisticRegressionModel-to-Predict-Placement-Status-of-Student/assets/118707073/d946cbc1-3e48-4072-a2f8-638d05d0152c">

<img src="https://github.com/ROHITJAIND/EX-04-Implementation-of-LogisticRegressionModel-to-Predict-Placement-Status-of-Student/assets/118707073/8ea856b4-6954-42a0-9837-72f0d6e6704c">

<img src="https://github.com/ROHITJAIND/EX-04-Implementation-of-LogisticRegressionModel-to-Predict-Placement-Status-of-Student/assets/118707073/8b0b67a9-c0ca-497a-b339-63284c8cc3b9">
<br>

**Accuracy:**&emsp;&emsp;&emsp;&emsp;&emsp;**Confusion Matrix:**&emsp;&emsp;&emsp;&emsp;&emsp;**Classification Report:**
<br>

<img src="https://github.com/ROHITJAIND/EX-04-Implementation-of-LogisticRegressionModel-to-Predict-Placement-Status-of-Student/assets/118707073/b843bb97-c5a7-48d7-8d83-a71eaa7d572a">

<img src="https://github.com/ROHITJAIND/EX-04-Implementation-of-LogisticRegressionModel-to-Predict-Placement-Status-of-Student/assets/118707073/b655b68f-0c41-44a4-b80c-c76d60159005">

<img src="https://github.com/ROHITJAIND/EX-04-Implementation-of-LogisticRegressionModel-to-Predict-Placement-Status-of-Student/assets/118707073/8260f1c9-eefa-4d64-ab41-c4afee548660">

### Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
