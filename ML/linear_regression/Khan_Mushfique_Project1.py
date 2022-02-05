""" Dependencies: python 3.9.7
                  matplotlib 3.5.0
                  numpy 1.21.2
                  openpyxl 0.46
                  pandas 1.3.5 """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def J(weight, design, target):
    """ Input: W: weight matrix 1,1   (D+1X1)
               X: design matrix 406,2 (NXD+1)
               t: target vector 406,1 (NX1)
        Output: gradient with respect to W  1,2 (1XD+1) 
        gradient of J with respect to W = 2W^T.X^T.X - 2t^T.X (from notes)
        (with respect to W)"""
    return 2*weight.T.dot(design.T.dot(design)) - 2*target.T.dot(design)

# Enter filename here
filepath = "proj1Dataset.xlsx"
df  = pd.read_excel(filepath)

rows = df['Weight'].shape[0]
features = np.array(df['Weight'])
# create matrix of shape N x D
features = np.reshape(features, (rows, 1))
X0 = np.ones((rows,1))

# create design matrix of shape N x D+1
design_matrix = np.hstack((features, X0))

t = df['Horsepower']
t = pd.DataFrame(t, columns=["Horsepower"])
t = t.fillna(int(df['Horsepower'].mean()))

# Closed form solution

# W = (X^T.X)^-1.X^T.t
W = (np.linalg.pinv(design_matrix))@t
Y = design_matrix@W

y_cf_plot = plt.scatter(df['Weight'], df['Horsepower'],c='red',marker='x')
y_cf_plot = plt.xlabel("Weight")
y_cf_plot = plt.ylabel("Horsepower")
y_cf_plot = plt.plot(df['Weight'], Y, c='b',label='Closed Form')
y_cf_plot = plt.legend(loc='upper right')
y_cf_plot = plt.title("Matlab's 'carbig' dataset")
y_cf_plot = plt.savefig("Closed Form.jpg")
plt.show()

# Gradient Descent Section

np.random.seed(44)
# Adjust learning rate (must be <10^-8) and 
# number of iterations gradient descent will run for
learning_rate = 0.0000000001
num_iterations = 5000

W0 = np.random.randn(2,1)
W_GD = W0
weightX = pd.DataFrame(design_matrix, columns=['Weight', 'W0'])

for i in range(num_iterations):
    W_temp = W_GD - learning_rate*(J(W_GD, weightX, t).T)
    W_GD = W_temp

Y_GD = design_matrix@W_GD
y_gd_plot = plt.scatter(df['Weight'], df['Horsepower'],c='red',marker='x')
y_gd_plot = plt.xlabel("Weight")
y_gd_plot = plt.ylabel("Horsepower")
y_gd_plot = plt.plot(df['Weight'], Y_GD, c='g',label='Gradient Descent')
y_gd_plot = plt.legend(loc='upper right')
y_gd_plot = plt.title("Matlab's 'carbig' dataset")
y_gd_plot = plt.savefig("Gradient Descent.jpg")
plt.show()