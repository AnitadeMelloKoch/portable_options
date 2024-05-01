import numpy as np
import matplotlib.pyplot as plt


class SVMEvaluator():
    @staticmethod
    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy
    
    @staticmethod
    def plot_contours(ax,
                      get_point, 
                      svm, 
                      x, 
                      y):
        xx, yy = SVMEvaluator.make_meshgrid(x, y)
        points = []
        predictions = []
        
        for _x in xx:
            for _y in yy:
                predictions.append(svm.predict(get_point(_x, _y)))
        
        points = np.array(points)
        points = points.reshape(xx.shape)
        
        out = ax.contourf(xx, yy, points, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        
        return out
    
    @staticmethod
    def plot_surface(x_linspace,
                     y_linspace,
                     svm,
                     ax):
        x, y = np.meshgrid(x_linspace, y_linspace)
        z = lambda x,y: (-svm.intercept_[0]-svm.coef_[0][0]*x-svm.coef_[0][1]*y)/svm.coef_[0][2]
        
        ax.plot_surface(x, y, z(x,y))
        ax.view_init(30, 60)
    
    
