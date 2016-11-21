from __future__ import division  # which forces / to adopt Python 3.x's behavior that always returns a float.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D


num_epochs_to_learn_representation = 7
convergence = 50


# this is the learning rate (a funtion of iteration). Later i shold be the epoch
# and we need to sample the training datasets p in {500, 250, 100, 50, 10, 1} and
# traing mentee on every settings.

def compute_n(i):
    n = 0
    if i < convergence:
        n = 0.01 - (0.005 * i / convergence)
    return n


def compute_eta_alpha(i, mode):
    alpha = 0
    # obedient
    if mode == "obedient":
        if i < num_epochs_to_learn_representation:
            alpha = 0.005 + (float(i / num_epochs_to_learn_representation) * 0.015)
            # print (type(alpha))

        elif i < convergence:
            alpha = 0.02 - (
                (i - num_epochs_to_learn_representation) / (convergence - num_epochs_to_learn_representation) * 0.005)

    # adamant
    elif mode == "adamant":
        if i < num_epochs_to_learn_representation:
            alpha = 0.04 - (i / num_epochs_to_learn_representation) * 0.033
        else:
            alpha = compute_n(i)

     #independent
    elif mode == "independent":
        alpha = 0.005 + (float(i / num_epochs_to_learn_representation) * 0.015)

    return alpha


def compute_eta_beta(i, mode):
    # obedient
    beta = 0
    if mode == "obedient":
        if i < convergence:
            beta = 0.04 - (i / convergence) * 0.035

    # adamant
    elif mode == "adamant":
        if i < convergence:
            beta = 0.01 - (i / convergence) * 0.008
    # independent
    elif mode == "independent":
        beta = 0
    return beta


def compute_eta_gamma(i, mode):
    # obedient
    gamma = 0
    if mode == "obedient":
        if i < convergence:
            gamma = 0.01 - (i / (num_epochs_to_learn_representation)) * 0.01
            if gamma < 0:
                gamma = 0
    # adamant
    elif mode == "adamant":
        if i < convergence:
            gamma = 0.005 - (i / (num_epochs_to_learn_representation)) * 0.005
            if gamma < 0:
                gamma = 0
    # independent
    elif mode == "independent":
        gamma=0

    return gamma


def plot_learning_rates():
    plt.interactive(False)


    n_list_obedient = []
    alpha_list_obedient = []
    beta_list_obedient = []
    gamma_list_obedient = []

    n_list = []
    alpha_list_adamant = []
    beta_list_adamant = []
    gamma_list_adamant = []

    for i in range(200):
        n = compute_n(i)
        alpha = compute_eta_alpha(i, 'obedient')
        beta = compute_eta_beta(i, 'obedient')
        gamma = compute_eta_gamma(i, 'obedient')

        n_list.append(n)
        alpha_list_obedient.append(alpha)
        beta_list_obedient.append(beta)
        gamma_list_obedient.append(gamma)

        n_list.append(n)
        alpha_list_adamant.append(compute_eta_alpha(i, 'adamant'))
        beta_list_adamant.append(compute_eta_beta(i, 'adamant'))
        gamma_list_adamant.append(compute_eta_gamma(i, 'adamant'))


        # print ("i = ", i, " and alpha= ", float(alpha))
        # print ("i = ", i, " and beta= ", beta)
        # print ("i = ", i, " and gamma= ", gamma)

    plt.plot(n_list, linewidth=3)
    plt.plot(alpha_list_obedient, linewidth=3)
    plt.plot(beta_list_obedient, linewidth=3)
    plt.plot(gamma_list_obedient, linewidth=3)
    plt.legend(["eta", "alpha * eta", "beta * eta", "gamma * eta"])

    plt.ylabel('obedient mode')
    plt.xlabel('epoch')
    plt.show()
    plt.close()

    plt.plot(n_list, linewidth=3)
    plt.plot(alpha_list_adamant, linewidth=3)
    plt.plot(beta_list_adamant, linewidth=3)
    plt.plot(gamma_list_adamant, linewidth=3)

    plt.legend(["eta", "alpha * eta", "beta * eta", "gamma * eta"])

    plt.ylabel('adamant mode')
    plt.xlabel('epoch')
    plt.show()
    plt.close()

if __name__ == "__main__":
    plot_learning_rates()
