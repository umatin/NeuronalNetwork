import numpy as np

def newNN(layers):
    ret = []
    for i in range(len(layers)-1):
        ret.append(np.random.rand(layers[i+1],layers[i] + 1)*2 - 1)
    return ret

def relu(x):
    if x < 0:
        return 0
    else:
        return x
relu_vec = np.vectorize(relu)

def relu_dash(x):
    if x < 0:
        return 0.01
    else:
        return 1
relu_dash_vec = np.vectorize(relu_dash)

def evalNN(nn, x):
    xn = x
    activations = []
    for w in nn:
        xn = np.append(xn, np.array([1]))
        activations.append(xn)
        xn = np.atleast_2d(w).dot(xn)
        xn = relu_vec(xn)
        
    activations.append(xn)
    return activations #Activations

def backprop_NN(nn,x,t):
    a = evalNN(nn,x)
    delta = []
    d1 = (a[-1] - t)
    d2 = relu_dash_vec(nn[-1].dot(a[-2]))
    d = d1*d2

    update = []
    update.insert(0, np.atleast_2d(d).T.dot(np.atleast_2d(a[-2])))
    
    delta.insert(0, d)
    for i in reversed(range(len(nn)-1)):
        d1 = nn[i+1].T.dot(delta[0])
        d2 = relu_dash_vec(nn[i].dot(a[i]))
        d2 = np.append(d2, np.array([0]))
        d = d1 * d2
        d = d[:-1]
        delta.insert(0, d)
        update.insert(0, np.atleast_2d(d).T.dot(np.atleast_2d(a[i])))
    return update

def train(x, y, nn, reps, lr):
    for _ in range(reps):
        for i in range(len(x)):
            update = backprop_NN(nn, x[i], y[i])
            for j in range(len(nn)):
                nn[j] = nn[j] - (update[j]*lr)

def test(x, y, nn):
    for i in range(len(x)):
        ret = evalNN(nn, x[i])
        print(x[i], ret[-1], y[i])


x = [np.array([-1,-1]), np.array([-1,1]), np.array([1,-1]), np.array([1,1])]
y = [np.array([0]), np.array([1]), np.array([1]), np.array([0])]
nn = newNN([2,10,10,1])
print(nn)
test(x, y, nn)
print("-----------")
train(x,y,nn,1000, 0.1)
test(x, y, nn)
print(nn)
#from sklearn.datasets import make_moons 
#X, y = make_moons(n_samples = 300, noise = 0.02, random_state = 417) 