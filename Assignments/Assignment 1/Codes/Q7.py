'''
Q7
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Main Functions
def P_x_theta(x, theta):
    if x >= 0:
        return theta * np.exp(-theta * x)
    else:
        return 0

def Theta_MLE(Xs):
    return Xs.shape[0] / np.sum(Xs)

def Plot_Prob_X_Fixed(x, theta_min, theta_max, n):
    thetas = np.linspace(theta_min, theta_max, n)
    P_x_theta_values = [P_x_theta(x, theta) for theta in thetas]
    plt.plot(thetas, P_x_theta_values)
    plt.xlabel('θ')
    plt.ylabel('P(x|θ)')
    plt.title('P(x|θ) vs θ for x = ' + str(x))
    plt.show()

def Plot_Prob_Theta_Fixed(theta, x_min, x_max, n):
    xs = np.array(list(np.linspace(x_min, 0, n)) + list(np.linspace(0, x_max, n)))
    P_x_theta_values = [P_x_theta(x, theta) for x in xs]
    plt.plot(xs, P_x_theta_values)
    plt.xlabel('x')
    plt.ylabel('P(x|θ)')
    plt.title('P(x|θ) vs x for θ = ' + str(theta))
    plt.show()

def Plot_Prob_Theta_Fixed_WithMLE(theta, x_min, x_max, n):
    xs = np.array(list(np.linspace(x_min, 0, n)) + list(np.linspace(0, x_max, n)))
    xs_MLE = np.linspace(0, x_max, n)
    theta_MLE = Theta_MLE(xs_MLE)
    P_x_theta_values = [P_x_theta(x, theta) for x in xs]
    P_x_theta_MLE_values = [P_x_theta(x, theta_MLE) for x in xs]

    plt.plot(xs, P_x_theta_values, label="θ = " + str(theta))
    plt.plot(xs, P_x_theta_MLE_values, label="θ_MLE = " + str(theta_MLE))
    plt.xlabel('x')
    plt.ylabel('P(x|θ)')
    plt.title('P(x|θ) vs x for θ = ' + str(theta) + " and θ_MLE = " + str(theta_MLE))
    plt.legend()
    plt.show()

def LogPlot_Prob_Theta_Fixed_WithMLE(theta, x_min, x_max, n):
    xs = np.array(list(np.linspace(x_min, 0, n)) + list(np.linspace(0, x_max, n)))
    xs_MLE = np.linspace(0, x_max, n)
    theta_MLE = Theta_MLE(xs_MLE)
    P_x_theta_values = np.log([P_x_theta(x, theta) for x in xs])
    P_x_theta_MLE_values = np.log([P_x_theta(x, theta_MLE) for x in xs])

    plt.plot(xs, P_x_theta_values, label="θ = " + str(theta))
    plt.plot(xs, P_x_theta_MLE_values, label="θ_MLE = " + str(theta_MLE))
    plt.xlabel('x')
    plt.ylabel('Ln(P(x|θ))')
    plt.title('Ln(P(x|θ)) vs x for θ = ' + str(theta) + " and θ_MLE = " + str(theta_MLE))
    plt.legend()
    plt.show()

# Driver Code
# Q7 (a) 
# θ = 1
Plot_Prob_Theta_Fixed(1, -10, 10, 100)
# x = 2, 0 <= θ <= 5
Plot_Prob_X_Fixed(2, 0, 5, 100)
# Q7 (c)
# θ = 1 and also θ = θ_MLE
Plot_Prob_Theta_Fixed_WithMLE(1, -10, 10, 100)
LogPlot_Prob_Theta_Fixed_WithMLE(1, -10, 10, 100)