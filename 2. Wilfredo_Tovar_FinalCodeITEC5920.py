# APPLIED PROGRAMMING ITEC5920 PROJECT
# Student: Wilfredo Alejandro Tovar Hidalgo / Student ID. 101100525
# Title: Simulator Neural Network Design in Python.
# Subject/Field: Artificial intelligence.

# NOTE: Depending on the integrated cross-platform you may need to close the windows to update the visualizations
# since it is a simulator I wanted to have all the graphics available for the analysis of the network layers.
# If you use Spyder as in class it will not be necessary to carry out any extra process.
# However, in case you use Pycharm you must close the windows to see a new progress visualization of the learning process.

# Depending on the complexity of the design and the number of nodes, the process may take approximately 5 minutes to complete.

#######    External libraries used for the project.     #####
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from tkinter import*


####     Creation of the artificial dataset        ####


root = Tk()
root.title("Input Section - Wilfredo Tovar's Neuronal Network Project")
root.geometry("640x640+0+0")
heading = Label(root, text="Input Section - Wilfredo Tovar's Neuronal Network Project", font=("arial", 16, "bold"), fg= "steelblue").pack()
heading = Label(root, text="", font=("arial", 16, "bold"), fg= "steelblue").pack()
heading = Label(root, text="***Recommendation***: Insert the Number of Nodes", font=("arial", 16, "bold"), fg= "darkred").pack()
heading = Label(root, text="between 300-600 for a better visualization.", font=("arial", 16, "bold"), fg= "darkred").pack()
label = Label(root, text="Insert Integer Number of Nodes:", font=("arial",12, "bold"), fg="black").place(x=10, y=200)
name = StringVar()
entry_box = Entry(root, textvariable=name, width=25, bg="lightgreen").place(x=280, y=210)


def do_it():

    return(int(name.get()))

work_close = Button(root, text="ENTER", width=30, height=4, bg="Lightblue", command=root.destroy).place(x=250, y=300)

root.mainloop()


####   Sample of Nodes and Classes of the Data Set    ####.


n = (do_it())
if n <= 0:
    exit()

p = 2

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.20)
Y = Y[:, np.newaxis]

plt.figure(1)
plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
plt.axis("equal")
plt.xticks([])
plt.yticks([])
plt.title("Original Topology")


####    NETWORK LAYER CLASS    ####


class neural_layer():

    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f

        self.b = np.random.rand(1, n_neur) * 2 - 1
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1


####    ACTIVATION FUNCTIONS    ####

##Function Sigmoide##

sigm = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x * (1 - x))

##Function ReLu##

relu = lambda x: np.maximum(0, x)

_x = np.linspace(-5, 5, 100)


####     Design and Construction of the Neural Network     ####

def create_nn(topology, act_f):

    nn = []

    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l + 1], act_f))

    return nn

####        TRAINING FUNCTION      ####

topology = [p, 4, 8, 1]

neural_net = create_nn(topology, sigm)

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: (Yp - Yr))


def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):
    out = [(None, X)]

    #### Forward Pass ####

    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b
        a = neural_net[l].act_f[0](z)

        out.append((z, a))

    if train:

    #### Backward Pass ####

        deltas = []
        for l in reversed(range(0, len(neural_net))):

            z = out[l + 1][0]
            a = out[l + 1][1]

            if l == len(neural_net) - 1:
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            _W = neural_net[l].W

    #### Gradient descent ####

            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr

    return out[-1][1]


train(neural_net, X, Y, l2_cost, 0.5)
print("")

#####         VISUALIZATION AND TEST      ####

import time
from IPython import display

neural_n = create_nn(topology, sigm)

loss = []

for i in range(2500):


    #####  NETWORK TRAINING PHASE #####

    pY = train(neural_n, X, Y, l2_cost, lr=0.05)

    if i % 25 == 0:

        ####  print(pY) <---------- IF YOU WANT TO VISUALIZE THE MATHEMATICAL PROCESS IN THE OUTPUT YOU CAN REMOVE THE COMMENT.

        loss.append(l2_cost[0](pY, Y))

        res = 50

        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)

        _Y = np.zeros((res, res))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), Y, l2_cost, train=False)[0][0]


        plt.figure(2)
        plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        plt.title("Visualization of Class Prediction by the Neuronal Network")
        plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
        plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
        plt.xlabel('The Color Area refers to influence sector predicted by the Neuronal Network')


        #plt.show()
        plt.figure(3)
        plt.title("Loss Curve for the Learning Rate")
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.plot(range(len(loss)), loss)
        time.sleep(0.5)
        display.clear_output(wait=True)
        plt.show()

#### THANK YOU VERY MUCH PROFESSOR MOHAMED ####