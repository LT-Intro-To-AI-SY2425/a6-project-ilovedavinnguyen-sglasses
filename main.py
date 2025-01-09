import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math

data = pd.read_csv("census-income.csv") #a6-project-ilovedavinnguyen-sglasses/
x = data["age"].values #AAGE
y = data["total person income"].values #PTOTVAL


model = LinearRegression().fit(x,y) 


plt.xlabel("Age")
plt.ylabel("total person income")

plt.scatter(x, y)
plt.plot(x, c="r", label="Line of Best Fit")
plt.legend()
plt.show()