import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Helvetica']})
rc('text', usetex=True)



def plt_surface3d(surrogate_problem, filename):

    # Make data.
    x1 = np.arange(-1, 1, 0.01)
    x2 = np.arange(-1, 1, 0.01)

    #R = np.sqrt(x1**2 + x2**2)
    #y = np.sin(R)
    X = None
    for i in x1:
        for j in x2:
            if X is None:
                X = [i, j]
            else:
                X = np.vstack((X,[i,j]))
    x1, x2 = np.meshgrid(x1, x2)
    y = surrogate_problem.evaluate(X, use_surrogate=True)[0]
    y1 = y[:,0]
    y2 = y[:,1]
    y3 = y[:,2]

    y1 = y1.reshape(np.shape(x1))
    # Plot the surface.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, y1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(filename + '_1.pdf', bbox_inches='tight')

    y2 = y2.reshape(np.shape(x1))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, y2, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(filename + '_2.pdf', bbox_inches='tight')

    y3 = y3.reshape(np.shape(x1))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1, x2, y3, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(filename + '_3.pdf', bbox_inches='tight')


def plt_surface2d(surrogate_problem, filename):

    # Make data.
    x1 = np.arange(-1, 1, 0.01)
    x2 = np.arange(-1, 1, 0.01)

    #R = np.sqrt(x1**2 + x2**2)
    #y = np.sin(R)
    X = None
    for i in x1:
        for j in x2:
            if X is None:
                X = [i, j]
            else:
                X = np.vstack((X,[i,j]))
    x1, x2 = np.meshgrid(x1, x2)
    y = surrogate_problem.evaluate(X, use_surrogate=True)[0]
    y1 = y[:,0]
    y2 = y[:,1]
    y3 = y[:,2]

    y1 = y1.reshape(np.shape(x1))
    y2 = y2.reshape(np.shape(x1))
    y3 = y3.reshape(np.shape(x1))
    # Plot the surface.
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.contour(x1,x2,y1,colors='red')
    plt.contour(x1,x2,y2,colors='blue')
    plt.contour(x1,x2,y3,colors='green')  
    plt.savefig(filename + '_contour.pdf', bbox_inches='tight')
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    plt.close()
