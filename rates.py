from __future__ import division  # which forces / to adopt Python 3.x's behavior that always returns a float.
import matplotlib.pyplot as plt

learning_the_representation_phase = 50
convergence = 80


# this is the learning rate (a funtion of iteration). Later i shold be the epoch
# and we need to sample the training datasets p in {500, 250, 100, 50, 10, 1} and
# traing mentee on every settings.

def compute_n(i):
    n = 0
    if i < convergence:
        n = 0.01 - (0.005 * i / convergence)
    return n


def compute_alpha(i, mode='obedient'):
    alpha = 0
    # obedient
    if mode == 'obedient':
        if i < learning_the_representation_phase:
            alpha = 0.005 + (float(i / learning_the_representation_phase) * 0.015)
            # print (type(alpha))

        elif i < convergence:
            alpha = 0.02 - (
                (i - learning_the_representation_phase) / (convergence - learning_the_representation_phase) * 0.005)


    # adamant
    if mode == 'adamant':
        if i < learning_the_representation_phase:
            alpha = 0.04 - (i / learning_the_representation_phase) * 0.033
        else:
            alpha = compute_n(i)

    return alpha


def compute_beta(i, mode='obedient'):
    # obedient
    beta = 0
    if mode == 'obedient':
        if i < convergence:
            beta = 0.04 - (i / convergence) * 0.035

    # adamant
    if mode == 'adamant':
            if i < convergence:
                beta = 0.01 - (i / convergence) * 0.008
    return beta


def compute_gamma(i, mode='obedient'):
    # obedient
    gamma = 0
    if mode == 'obedient':
        if i < convergence:
            gamma = 0.01 - (i / (learning_the_representation_phase)) * 0.01
            if gamma < 0:
                gamma = 0
    # adamant
    if mode == 'adamant':
        if i < convergence:
            gamma = 0.005 - (i / (learning_the_representation_phase)) * 0.005
            if gamma < 0:
                gamma = 0

    return gamma


# n_list_obedient = []
# alpha_list_obedient = []
# beta_list_obedient = []
# gamma_list_obedient = []
#
# n_list = []
# alpha_list_adamant = []
# beta_list_adamant = []
# gamma_list_adamant = []
#
#
# for i in range(100):
#     n = compute_n(i)
#     alpha = compute_alpha(i, 'obedient')
#     beta = compute_beta(i, 'obedient')
#     gamma = compute_gamma(i, 'obedient')
#
#     n_list.append(n)
#     alpha_list_obedient.append(alpha)
#     beta_list_obedient.append(beta)
#     gamma_list_obedient.append(gamma)
#
#     n_list.append(n)
#     alpha_list_adamant.append(compute_alpha(i, 'adamant'))
#     beta_list_adamant.append(compute_beta(i, 'adamant'))
#     gamma_list_adamant.append(compute_gamma(i, 'adamant'))
#
#
#     # print ("i = ", i, " and alpha= ", float(alpha))
#     # print ("i = ", i, " and beta= ", beta)
#     # print ("i = ", i, " and gamma= ", gamma)
#
#
#
# plt.plot(n_list)
# plt.plot(alpha_list_obedient)
# plt.plot(beta_list_obedient)
# plt.plot(gamma_list_obedient)
#
# plt.ylabel('obedient mode')
# plt.show()
# plt.close()
#
# plt.plot (n_list)
# plt.plot(alpha_list_adamant)
# plt.plot(beta_list_adamant)
# plt.plot(gamma_list_adamant)
#
# plt.ylabel('adamant mode')
# plt.show()
# plt.close()
