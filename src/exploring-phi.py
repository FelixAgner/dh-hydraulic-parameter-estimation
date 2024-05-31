# script for inspecting the data matrix phi and its' properties.
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import load_data_matrices

def run():
    # load phi for model C
    phi_E, y_E = load_data_matrices("C", "exciting")
    phi_R, y_R = load_data_matrices("C", "realistic")
    n_E = phi_E.shape[0]
    n_R = phi_R.shape[0]

    print(f"""Data matrix shapes: 
          Exciting: {phi_E.shape}, 
          Realistic: {phi_R.shape}""")

    # calculate condition number for phi^T phi
    PPT_E = phi_E.T @ phi_E / n_E
    PPT_R = phi_R.T @ phi_R / n_R

    # make inverses
    PPT_E_inv = np.linalg.inv(PPT_E)
    PPT_R_inv = np.linalg.inv(PPT_R)

    if False:
        cond_E = np.linalg.cond(PPT_E)
        cond_R = np.linalg.cond(PPT_R)

        print(f"""Condition number for phi^T phi for exciting data:
            Exciting: {cond_E}
            Realistic: {cond_R}""")
    
    # heatmap the two covariance matrices in 2x2subfigures
    fig, (ax, ax_off) = plt.subplots(2,2)
    ax[0].imshow(PPT_E, cmap='hot', interpolation='nearest')
    ax[0].set_title("Covariance matrix for exciting data")
    ax[1].imshow(PPT_R, cmap='hot', interpolation='nearest')
    ax[1].set_title("Covariance matrix for realistic data")
    ax_off[0].imshow(PPT_E[60*2:60*3, 60*3:60*4], cmap='hot', interpolation='nearest')
    ax_off[0].set_title("Off-diagonal blocks for exciting data")
    ax_off[1].imshow(PPT_R[60*2:60*3, 60*3:60*4], cmap='hot', interpolation='nearest')
    ax_off[1].set_title("Off-diagonal blocks for realistic data")

    # make heatmaps of inverse matrices
    fig_inv, (ax_inv, ax_off_inv) = plt.subplots(2,2)
    ax_inv[0].imshow(PPT_E_inv, cmap='hot', interpolation='nearest')
    ax_inv[0].set_title("Inverse covariance matrix for exciting data")
    ax_inv[1].imshow(PPT_R_inv, cmap='hot', interpolation='nearest')
    ax_inv[1].set_title("Inverse covariance matrix for realistic data")
    ax_off_inv[0].imshow(PPT_E_inv[60*2:60*3, 60*3:60*4], cmap='hot', interpolation='nearest')
    ax_off_inv[0].set_title("Off-diagonal blocks for exciting data")
    ax_off_inv[1].imshow(PPT_R_inv[60*2:60*3, 60*3:60*4], cmap='hot', interpolation='nearest')
    ax_off_inv[1].set_title("Off-diagonal blocks for realistic data")

    plt.show()



    


if __name__ == "__main__":
    run()