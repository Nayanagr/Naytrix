import numpy as np

from linear_regression import LinearRegression

model = LinearRegression(itr=1000,l_r=0.001)
sample_x = np.array([1,2,3,4,5,6,7,8,9,10])
sample_x=sample_x.reshape(-1,1)
print(sample_x.shape)
sample_y = np.array([1,4,6,8,10,12,14,16,18,20])
test_array = np.array([5,6]).reshape(-1,1)

model.fit(sample_x,sample_y)
print(model.predict(test_array))
