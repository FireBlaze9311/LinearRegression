from sklearn.datasets import load_boston
import numpy as np

dataset = load_boston()

class LinearReg:
    
    def compute_slope(self,x,y, x_mean, y_mean):
        frac1 = sum([(x[i]-x_mean)*(y[i]-y_mean) for i in range(len(x))])
        frac2 = sum([(x[i]-x_mean)**2 for i in range(len(x))])
        slope = frac1/frac2
        return slope


    def compute_intercept(self,x_mean, y_mean, slope):
        b = y_mean-slope*x_mean
        return b

    def fit(self,x, y):
        self.slope = self.compute_slope(x, y,np.mean(x), np.mean(y))
        self.intercept = self.compute_intercept(np.mean(x), np.mean(y), self.slope)

    def predict(self,y):
        x= y-self.intercept/self.slope
        return x

x=[1, 2, 3, 4, 5]
y=[1,2,3,4,5]

reg = LinearReg()
reg.fit(x,y)

print(reg.predict(500))


