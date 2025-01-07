from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
census_income_kdd = fetch_ucirepo(id=117) 
  
# data (as pandas dataframes) 
X = census_income_kdd.data.features 
y = census_income_kdd.data.targets 
  
# metadata 
print(census_income_kdd.metadata) 
  
# variable information 
print(census_income_kdd.variables) 