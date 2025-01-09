import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math

data = pd.read_csv("census-income.csv")
x = data["Age"].values
y = data["Blood Pressure"].values
