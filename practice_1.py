import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from warnings import filterwarnings
filterwarnings('ignore')

def criteria_plot():
    plt.figure(figsize=(6,4))
    xx = np.linspace(0,1,50)
    plt.plot(xx, [2*x*(1-x) for x in xx], label='gini')
    plt.plot(xx, [4*x*(1-x) for x in xx], label='2*gini')
    plt.plot(xx, [-x* np.log2(x) - (1-x)* np.log2(1-x) for x in xx], label='entropy')
    plt.plot(xx, [1 - max(x, 1-x) for x in xx], label='missclass')
    plt.plot(xx, [2 -2*max(x, 1 - x) for x in xx], label='2*missclass')
    plt.xlabel('p+')
    plt.ylabel('criterion')
    plt.title('Criteria of quality as a function of p+ (binary classification)')
    plt.legend()
    plt.show()

def get_grid(data):
    x_min, x_max =data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

def tree_graph_to_png(tree, feature_names, png_file_to_save) :
    tree_str = export_graphviz(tree, feature_names=feature_names,
                                     filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(png_file_to_save)

def main():
    #first class
    np.random.seed(17)
    train_data = np.random.normal(size=(100,2))
    train_labels = np.zeros(100)

    #second class
    train_data   = np.r_[train_data, np.random.normal(size=(100,2), loc=2)]
    train_labels = np.r_[train_labels, np.ones(100)]

    plt.figure(figsize=(10,8))
    plt.scatter(train_data[:,0], train_data[:, 1],
                c=train_labels,
                s=100, cmap='autumn',
                edgecolors='black',
                linewidths=1.5)

    plt.plot(range(-2,5), range(4,-3,-1));
    plt.show()

    clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)
    clf_tree.fit(train_data, train_labels)
    xx, yy = get_grid(train_data)
    predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.pcolormesh(xx, yy, predicted, cmap='autumn')
    plt.scatter(train_data[:, 0], train_data[:, 1],
    c = train_labels,
    s = 100, cmap = 'autumn',
    edgecolors = 'black',
    linewidths = 1.5)
    plt.show()

    tree_graph_to_png(tree=clf_tree, feature_names=['x1', 'x2'], png_file_to_save='dt1.png')



main()



