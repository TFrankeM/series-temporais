import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv("us_change.csv")

X = df[['Income', 'Production', 'Savings', 'Unemployment']]
y = df['Consumption']

X = (X - X.mean()) / X.std()
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())
# Utilizando apenas conhecimento do passado

df_lagged = df.copy()
df_lagged[['Income', 'Production', 'Savings', 'Unemployment','Consumption']] = df[['Income', 'Production', 'Savings', 'Unemployment', 'Consumption']].shift(1)

df_lagged = df_lagged.dropna()

X = df_lagged[['Income', 'Production', 'Savings', 'Unemployment', 'Consumption']]
y = df['Consumption']
y = y[1:]


X_standardized = (X - X.mean()) / X.std()


X_standardized = sm.add_constant(X_standardized)


model2 = sm.OLS(y, X_standardized).fit()


print(model2.summary())