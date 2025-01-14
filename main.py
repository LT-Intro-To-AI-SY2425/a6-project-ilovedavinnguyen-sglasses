import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import math
import statistics

data = pd.read_csv("census_data.csv")

x = data[["age", "education", "marital"]].values
y = data["income"].values

scaler = StandardScaler().fit(x)
x = scaler.transform(x)

meanfraccorr_sterr = []

psi = 0.05
for j in range(17):
    psi += 0.05
    temp_per = []
    for i in range(5):
        xTraining, xTest, yTraining, yTest = train_test_split(x,y,test_size=psi)
        model = linear_model.LogisticRegression().fit(xTraining,yTraining)
        
        correct_cases = 0
        for index in range(len(xTest)):
            reshapedx = xTest[index]
            reshapedx = reshapedx.reshape(-1,3)
            if (model.predict(reshapedx) >= 0.5 and yTest[index] == 1) or (model.predict(reshapedx) < 0.5 and yTest[index] == 0):
                correct_cases += 1
        corr_per = 100*correct_cases/len(yTest)
        temp_per.append(corr_per)
    meanfraccorr_sterr.append([statistics.mean(temp_per),(statistics.stdev(temp_per)/math.sqrt(len(xTest)))])
    
opt_fraccorr = 0
opt_sterr = 100
opt_index = 0
iteration = 0
for a,b in meanfraccorr_sterr:
    if (opt_fraccorr < a) and (opt_sterr > b):
        opt_fraccorr = a
        opt_sterr = b
        opt_index = iteration
    iteration += 1
optimal_split = (0.1)+(opt_index*0.05)

t = np.array([10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90])
c = []
e = []
for pair in meanfraccorr_sterr:
    a,b = pair
    c.append(a)
for pair in meanfraccorr_sterr:
    a,b = pair
    e.append(b)
fig, graph = plt.subplots(2)
graph[0].scatter(t, c)
graph[0].set_xlabel("% of Data Used for Testing")
graph[0].set_ylabel("Mean % of Cases Correct")
graph[1].scatter(t, e)
graph[1].set_xlabel("% of Data Used for Testing")
graph[1].set_ylabel("Standard Error of % Cases Correct")

xTraining, xTest, yTraining, yTest = train_test_split(x,y,test_size=optimal_split)
model = linear_model.LogisticRegression().fit(xTraining,yTraining)
correct_cases = 0
for index in range(len(xTest)):
    reshapedx = xTest[index]
    reshapedx = reshapedx.reshape(-1,3)
    if (model.predict(reshapedx) >= 0.5 and yTest[index] == 1) or (model.predict(reshapedx) < 0.5 and yTest[index] == 0):
         correct_cases += 1
corr_per = 100.0*correct_cases/len(yTest)
print(f"Optimal Percent for Training Data: {round((optimal_split*100),4)}%")
print(f"Percentage of Cases Correct at Optimal Split: {corr_per}")
plt.tight_layout()
plt.show()

