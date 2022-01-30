import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filepath = "proj1Dataset.xlsx"
df  = pd.read_excel(filepath)

rows = df['Weight'].shape[0]
features = np.array(df['Weight'])
# create matrix of shape N x D
features = np.reshape(features, (rows, 1))
X0 = np.ones((rows,1))

# create design matrix of shape N x D+1
design_matrix = np.hstack((features, X0))

pseudoinverse = np.linalg.pinv(design_matrix)

t = df['Horsepower']
t = pd.DataFrame(t, columns=["Horsepower"])
t = t.fillna(0)

W = np.matmul(pseudoinverse, t)
Y = design_matrix@W

plt.scatter(df['Weight'], df['Horsepower'],c='red',marker='x')
plt.xlabel("Weight")
plt.ylabel("Horsepower")
plt.plot(df['Weight'], Y, c='b',label='Closed Form')
plt.legend(loc='upper right')
plt.title("Matlab's 'carbig' dataset")
plt.savefig("Closed Form.jpg")