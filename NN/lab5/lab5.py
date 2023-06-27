import pandas as pd
dataset = pd.read_csv('iris.csv')
print(dataset)

# select only column SepalLengthCm
print(dataset['SepalLengthCm'])

# select columns SepalLengthCm and SepalWidthCm
print(dataset[['SepalLengthCm', 'SepalWidthCm']])

print(dataset.SepalLengthCm)
print(dataset.iloc[:, 0])

print(dataset[0:10:2])

print(dataset.loc[0:9:2])
print(dataset.iloc[0:9:2])

print(dataset[:10:2]['SepalLengthCm'])
print(dataset['SepalLengthCm'][:10:2])

print(dataset.loc[lambda df:df.SepalLengthCm > 5, :])

dataset['SepalLengthCm'][:10:].values


import seaborn as sns
import matplotlib.pyplot as plt

# set plot style
sns.set(style="ticks")
sns.set_palette("husl")

# create plots over all dataset; for subset use iloc indexing
sns.pairplot(dataset, hue="Species")

# display plots using matplotlib
plt.show()

# split data into input (X - select the first four columns) and output (y - select last column)
X = dataset.iloc[:, :4].values
y = dataset.iloc[:, -1].values


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
# transform string labels into number values 0, 1, 2
y1 = encoder.fit_transform(y)

# transform number values into vector representation
Y = pd.get_dummies(y1).values

import torch
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)


import torch

model = torch.nn.Sequential(
    torch.nn.Linear(4, 10),
    torch.nn.Tanh(),
    torch.nn.Linear(10, 3),
# TODO: add dense layer with 10, 8 neurons and tanh activation function
# TODO: add dense layer with 8, 6 neurons and tanh activation function
# TODO: add dense layer with 6, 3 neurons and softmax activation function
)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    preds = model(X_train)
    loss = criterion(preds, y_train.argmax(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Loss:", loss.detach().item(), "accuracy:", (y_train.argmax(-1) == preds.argmax(-1)).sum().item()/len(y_train))

y_pred = model(X_test)

import numpy as np

y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred.detach().numpy(),axis=1)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class))
