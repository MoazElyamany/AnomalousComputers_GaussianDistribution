import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Load data
Feature_list=['CPU usage(percentage)','Network traffic(bytes/second)','Disk usage(percentage)',
'System logs frequency(EPS)','Power consumption(watts)','Memory usage(percentage)','latency(ms)of server response',
'DNS query rate(QPM)','Hardware temperature(degrees Celsius)','Fan speed(RPM)','throughput(mb/s)']
def load_data():
    X_train = np.load("X_training.npy")
    X_val = np.load("X_validation.npy")
    y_val = np.load("y_validation.npy")
    print(f"# Dimension of The Training Data : \n {X_train.shape} \n ")
    print(f"# First two elements of X training : \n {X_train[:2]} \n ")
    print(f"# First two elements of X validation : \n {X_val[:2]} \n ")
    print(f"# First two elements of y validation : \n {y_val[:2]} \n ")
    # visualize histograms of each feature
    fig, axes = plt.subplots(2, 5, figsize=(15, 5), constrained_layout = True)
    for i, ax in enumerate(axes.flat):
        ax.hist(X_train[:,i], density= True)
        # Display the label above the image
        ax.set_title(Feature_list[i], fontsize=10)
    fig.suptitle("Distribution of features", fontsize=20)
    plt.show()
    return X_train, X_val, y_val
X_train, X_val, y_val = load_data()


# Estimate the Gaussian parameters
def Gaussian_parameters(X):
    m, n = X.shape
    mu = 1 / m * np.sum(X, axis=0)
    var = 1 / m * np.sum((X - mu) ** 2, axis=0)
    return mu, var
mu, var = Gaussian_parameters(X_train)


# Computes the probability
def Gaussian_probability(X, mu, var):
    p = 1 / (2 * np.pi * var) ** 0.5 * np.exp(-(X - mu) ** 2 / (2 * var))
    p = np.prod(p, axis=1, keepdims=True).flatten()
    return p
p = Gaussian_probability(X_train, mu, var)
p_val = Gaussian_probability(X_val, mu, var)


# Find the best threshold
def select_threshold(y_val, p_val):
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = (p_val < epsilon)
        tp = np.sum((predictions == 1) & (y_val == 1))
        fn = np.sum((predictions == 0) & (y_val == 1))
        fp = sum((predictions == 1) & (y_val == 0))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2 * prec * rec / (prec + rec)
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon, best_F1
best_epsilon, best_F1 = select_threshold(y_val, p_val)


# Visualize the fit
def Visualize():
    # Print the Gaussian parameters
    print('# Mean of each feature: \n', mu, '\n')
    print('# Variance of each feature: \n', var, '\n')
    print('# Best epsilon found using cross-validation: %e' % best_epsilon, '\n')
    print('# Best F1 on Cross Validation Set:  %f' % best_F1, '\n')
    print('# Number of Anomalies found: %d' % sum(p < best_epsilon), '\n')
    # Visualize the 3D fit
    outliers = p < best_epsilon
    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[:, 8], X_train[:, 4], X_train[:, 6],c='blue', marker='o')
    ax.scatter(X_train[outliers, 8], X_train[outliers, 4], X_train[outliers, 6],c='red', marker='o', s=70, edgecolors='red')
    ax.set_xlabel(Feature_list[8]);ax.set_ylabel(Feature_list[4]);ax.set_zlabel(Feature_list[6], fontsize=8)
    plt.show()
Visualize()


# Predict condition for new computers
def predict_condition(X_new):
    p_new = Gaussian_probability(X_new, mu, var)
    if p_new < best_epsilon :
        print("The new computer is anomalous \n")
    else :
        print("The new computer is normal \n")
X_new1 = np.array([[ 14.65311366e+00,  13.36897016e+01,  13.75528453e+01, 1.19334974e+01
  ,17.75640883e+00,  11.54359213e+01, 12.21367520e+01,  16.22412257e+00,
  12.88402408e+00 , 13.34933534e+00 , 1.73513724e+01]])
x_new2 = np.array([[ 1.79813538, -10.24806381,  18.26868719, -18.70913213,  -9.47755057,
   11.52797166,  -8.91917727,  17.23817848,  -3.28432466,  24.18311927,
    5.28667084]])
predict_condition(X_new1)
predict_condition(x_new2)