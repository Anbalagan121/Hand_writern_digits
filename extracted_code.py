pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
(x_train, y_train),(x_test,y_test)=mnist.load_data()
print("Training Images:",x_train.shape)
print("Training Labels:",y_train.shape)
print("Testing Images:",x_test.shape)
print("Testing Labels:",y_test.shape)
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.show()
unique, count=np.unique(y_train, return_counts=True)
plt.figure(figsize=(8,5))
sns.barplot(x=unique, y=count)
plt.title("Digit Distribution")
plt.xlabel("Digits")
plt.ylabel("count")
plt.show()
x_train_flat=x_train.reshape(60000,784)
x_test_flat=x_test.reshape(10000,784)
print(x_train_flat.shape)
x_train_flat=x_train_flat/255.0
x_test_flat=x_test_flat/255.0
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=100)
lr.fit(x_train_flat, y_train)
y_pred_lr=lr.predict(x_test_flat)
acc_lr=accuracy_score(y_test,y_pred_lr)
print("LogisticRegression Accuracy:",acc_lr)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_flat,y_train)
y_pred_knn=knn.predict(x_test_flat)
acc_knn=accuracy_score(y_test,y_pred_knn)
print("KNN Accuracy:",acc_knn)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(x_train_flat,y_train)
y_pred_rf=rf.predict(x_test_flat)
acc_rf=accuracy_score(y_test,y_pred_rf)
print("Random Forest Accuracy :",acc_rf)
from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train_flat, y_train)
y_pred_svm=svm.predict(x_test_flat)
acc_svm=accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy :", acc_svm)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
x_train_cnn=x_train.reshape(-1,28,28,1)/255.0
x_test_cnn=x_test.reshape(-1,28,28,1)/255.0
cnn=Sequential()
cnn.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
        
    )
)
cnn.add(
    MaxPooling2D(2,2)
)
cnn.add(Flatten())
cnn.add(
    Dense(
        128,
        activation='relu'
    )
)
cnn.add(
    Dense(
        10,
        activation='softmax'
    )
)
cnn.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history=cnn.fit(
    x_train_cnn,
    y_train,
    epochs=5,
    validation_data=(x_test_cnn,y_test)
)
loss, acc_cnn =cnn.evaluate(
    x_test_cnn,
    y_test
)
print('cnn Accuracy :', acc_cnn)
results=pd.DataFrame({
    'Model':[
        'Logistic Regression',
        'KNN',
        'Random Forest',
        'SVM',
        'CNN'
    ],
    'Accuracy':[
        acc_lr,
        acc_knn,
        acc_rf,
        acc_svm,
        acc_cnn
    ]
}
    )
print(results)
plt.figure(figsize=(8,5))
sns.barplot(
    x='Model',
    y='Accuracy',
    data=results
)
plt.xticks(rotation=30)
plt.title('Model Comparison')
plt.show()
from sklearn.metrics import confusion_matrix
y_pred=cnn.predict(x_test_cnn)
y_pred=np.argmax(y_pred, axis=1)
cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
