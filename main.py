from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression 

# fetch dataset 
census_income_kdd = fetch_ucirepo(id=117) 

# data (as pandas dataframes) 
x = census_income_kdd.data.features 
y = census_income_kdd.data.targets 
  
sklearn.linear_model.LinearRegression().fit(x, y)