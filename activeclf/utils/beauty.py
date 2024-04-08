# -------------------------------------------------- #
# Plot utilities
# for the Active Learning Classifier
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from typing import Tuple, Callable

from .misc import get_space_lims, make_meshgrid


# --- PLOT FUNCTIONS ---#

def plot_active_learning_cycle(feature_space: Tuple[np.ndarray,np.ndarray,np.ndarray], 
                               clfModel: Callable, 
                               idxs: list[int], 
                               new_idxs: list[int]) -> None:
    
    # Buildi the complete feature space (visualization)
    X0, X1, y = feature_space
    incr = '10%'
    x_min, x_max = get_space_lims(coord=X0, incr=incr)
    y_min, y_max = get_space_lims(coord=X1, incr=incr)
    xx, yy = make_meshgrid(x=X0, y=X1, incr=incr, delta=.1)

    # Classes specification and color grading
    try:
        n_classes = len(np.unique(y[idxs]))
    except:
        n_classes = len(np.unique(y.iloc[idxs]))
    CMAPS = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']

    # Build the feature space PDF and Entropy
    Z = clfModel.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # Z = MinMaxScaler().fit_transform(X=_Z)
    H = scipy.stats.entropy(pk=Z, axis=1)
    Z = Z.reshape((xx.shape[0], xx.shape[1], -1))

    # plots
    fig, ax = get_axes(n_classes+2,3)

    # class-wise PDFs
    for i in range(n_classes):
        # plot the initial data
        alphas = get_alphas(Z=Z[:,:,i])
        ax[i].scatter(X0[new_idxs], X1[new_idxs], edgecolor='0.', facecolor='none', s=50, zorder=4)
        surf = ax[i].imshow(Z[:,:,i], 
                            cmap=CMAPS[i], 
                            extent=(x_min, x_max, y_min, y_max), 
                            vmin=.0, vmax=1.,
                            origin="lower", 
                            aspect='auto', 
                            alpha=alphas.reshape(xx.shape))
        cbar = fig.colorbar(surf,ax=ax[i], format='%1.1f')
        cs = ax[i].contour(xx, yy, Z[:,:,i]-.501, 
                           colors='0.', levels=4, norm='linear', zorder=3)
        ax[i].set_title(f'pdf class-{i}')
        ax[i].set_xlabel('f0')
        ax[i].set_ylabel('f1')
        ax[i].set_xticks(())
        ax[i].set_yticks(())

    # class prediction plot
    ax[n_classes].scatter(X0[new_idxs], X1[new_idxs], edgecolor='0.', facecolor='none', s=50, zorder=4)
    ax[n_classes].scatter(X0[idxs], X1[idxs], c=y[idxs], edgecolor='0.', s=50, zorder=4)
    for j in range(n_classes):
        alphas = get_alphas(Z=Z[:,:,j], scale=True)
        ax[n_classes].imshow(Z[:,:,j], 
                             cmap=CMAPS[j], 
                             extent=(x_min, x_max, y_min, y_max), 
                             vmin=.0, vmax=1.,
                             origin="lower", 
                             aspect='auto', 
                             alpha=alphas.reshape(xx.shape))
    ax[n_classes].set_title(f'cls prediction')
    ax[n_classes].set_xlabel('f0')
    ax[n_classes].set_ylabel('f1')
    ax[n_classes].set_xticks(())
    ax[n_classes].set_yticks(())

    # entropy plot
    ax[n_classes+1].scatter(X0[new_idxs], X1[new_idxs], edgecolor='0.', facecolor='none', s=50, zorder=4)
    surf = ax[n_classes+1].imshow(H.reshape(xx.shape), 
                           extent=(x_min, x_max, y_min, y_max),
                           cmap='plasma', origin="lower", aspect='auto', alpha=1.)
    cbar = fig.colorbar(surf,ax=ax[n_classes+1], format='%1.2f')
    ax[n_classes+1].contour(xx, yy, H.reshape(xx.shape), 
                            levels=3, colors='0.', norm='linear', zorder=4)
    ax[n_classes+1].set_title(f'entropy')
    ax[n_classes+1].set_xlabel('f0')
    ax[n_classes+1].set_ylabel('f1')
    ax[n_classes+1].set_xticks(())
    ax[n_classes+1].set_yticks(())

    fig.tight_layout()


def plot_classification(feature_space: Tuple[np.ndarray,np.ndarray,np.ndarray], 
                        clfModel: Callable) -> None:
    
    # Buildi the complete feature space (visualization)
    X0, X1, y = feature_space
    incr = '10%'
    x_min, x_max = get_space_lims(coord=X0, incr=incr)
    y_min, y_max = get_space_lims(coord=X1, incr=incr)
    xx, yy = make_meshgrid(x=X0, y=X1, incr=incr, delta=.1)

    # Classes specification and color grading
    n_classes = len(np.unique(y))
    CMAPS = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']

    # Build the feature space PDF and Entropy
    Z = clfModel.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    H = scipy.stats.entropy(pk=Z, axis=1)
    Z = Z.reshape((xx.shape[0], xx.shape[1], -1))

    # plots
    fig, ax = get_axes(n_classes+2,3)

    for i in range(n_classes):
        # plot the initial data
        alphas = get_alphas(Z=Z[:,:,i])
        ax[i].scatter(X0, X1, c='0.', marker='.', s=50, zorder=4)
        surf = ax[i].imshow(Z[:,:,i], 
                    cmap=CMAPS[i], 
                    extent=(x_min, x_max, y_min, y_max), 
                    vmin=.0, vmax=1.,
                    origin="lower", 
                    aspect='auto', 
                    alpha=alphas.reshape(xx.shape))
        cbar = fig.colorbar(surf,ax=ax[i])
        cs = ax[i].contour(xx, yy, Z[:,:,i]-.50001, 
                           colors='0.', levels=4, norm='linear', zorder=3)
        ax[i].set_title(f'pdf class-{i}')
        ax[i].set_xlabel('f0')
        ax[i].set_ylabel('f1')
        ax[i].set_xticks(())
        ax[i].set_yticks(())

    ax[n_classes].scatter(X0, X1, c='0.', marker='.', s=50, zorder=4)
    ax[n_classes].scatter(X0, X1, c=y, edgecolor='0.', s=50, zorder=4)
    for j in range(n_classes):
        alphas = get_alphas(Z=Z[:,:,j], scale=True)
        ax[n_classes].imshow(Z[:,:,j], 
                             cmap=CMAPS[j], 
                             extent=(x_min, x_max, y_min, y_max), 
                             vmin=.0, vmax=1.,
                             origin="lower", 
                             aspect='auto', 
                             alpha=alphas.reshape(xx.shape))
    ax[n_classes].set_title(f'cls prediction')
    ax[n_classes].set_xlabel('f0')
    ax[n_classes].set_ylabel('f1')
    ax[n_classes].set_xticks(())
    ax[n_classes].set_yticks(())

    ax[n_classes+1].scatter(X0, X1, c='0.', marker='.', s=50, zorder=4)
    surf = ax[n_classes+1].imshow(H.reshape(xx.shape), 
                           extent=(x_min, x_max, y_min, y_max),
                           cmap='plasma', origin="lower", aspect='auto', alpha=1.)
    cbar = fig.colorbar(surf,ax=ax[n_classes+1], format='%1.2f')
    ax[n_classes+1].contour(xx, yy, H.reshape(xx.shape), 
                            levels=3, colors='0.', norm='linear', zorder=4)
    ax[n_classes+1].set_title(f'entropy')
    ax[n_classes+1].set_xlabel('f0')
    ax[n_classes+1].set_ylabel('f1')
    ax[n_classes+1].set_xticks(())
    ax[n_classes+1].set_yticks(())

    fig.tight_layout()


def plot_simple_al_output(X: Tuple[np.ndarray, np.ndarray], 
                          Z: np.ndarray, 
                          new_idxs: list[int],
                          minmaxScaling: bool=True) -> None:
    
    CMAPS = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']
    
    X0, X1 = X

    # compute Prob distribution
    if minmaxScaling:
        Z = MinMaxScaler().fit_transform(X=Z)

    # compute Entropy values
    H = scipy.stats.entropy(pk=Z, axis=1)

    n_classes = Z.shape[1]
    fig, ax = get_axes(2, 2)

    for i in range(n_classes):
        sizes = (Z[:,i] * (H  - max(H)) * -1) + 0.005
        alphas = get_alphas(Z=Z[:,i], scale=True)
        ax[0].scatter(X0[new_idxs], X1[new_idxs], 
                      edgecolor='0.', facecolor='none', 
                      s=50, zorder=2)

        ax[0].scatter(X0, X1, c=Z[:,i], cmap=CMAPS[i], 
                      alpha=alphas, s=sizes*70, 
                      marker='s', zorder=1)

    ax[1].scatter(X0[new_idxs], X1[new_idxs], 
                  edgecolor='0.', facecolor='none', s=50, zorder=2)
    ax[1].scatter(X0, X1, c=H, cmap='plasma', s=H*70, marker='s', zorder=1)

    for i in range(2):
        ax[i].set_xlabel('f0')
        ax[i].set_ylabel('f1')
        ax[i].set_xticks(())
        ax[i].set_yticks(())

    fig.tight_layout()


def plot_entropy3D(X, Z, decimals, scaling=True):
    fig = plt.figure(figsize=(4,4), dpi=200)
    ax = fig.add_subplot(projection='3d')

    X0,X1 = X
    if scaling:
        Z = MinMaxScaler().fit_transform(X=Z)
    H = np.around(scipy.stats.entropy(pk=Z, axis=1), decimals=decimals)

    ax.scatter(
        X0, X1, H,
        c=H,
        cmap='plasma',
        s=H*70,
        alpha=1.,
        edgecolor='0.',
    )

    ax.set_xlabel('f0')
    ax.set_ylabel('f1')
    ax.set_zlabel('H')
    ax.set_xticks(())
    ax.set_yticks(())


    ax.set_proj_type('ortho')

    ax.azim = -145
    ax.dist = 10
    ax.elev = 7

    fig.tight_layout()

# --- ////////////// ---#


# --- PLOT UTILITIES ---#

def get_alphas(Z: np.ndarray, scale: bool=False) -> np.ndarray:
    alphas = (Z.ravel() - Z.ravel().min()) / (Z.ravel().max() - Z.ravel().min())
    if scale:
        for i,av in enumerate(alphas):
            if av <= .50001:
                alphas[i] = 0.
            else:
                pass
    return alphas


def get_axes(plots: int, 
             max_col: int =2, 
             fig_frame: tuple =(3.3,3.), 
             res: int =200):
    """Define Fig and Axes objects.

    :param plots: number of plots frames in a fig.
    :type plots: int
    :param max_col: number of columns to arrange the frames, defaults to 2
    :type max_col: int, optional
    :param fig_frame: frame size, defaults to (3.3,3.)
    :type fig_frame: tuple, optional
    :param res: resolution, defaults to 200
    :type res: int, optional
    :return: fig and axes object from matplotlib.
    :rtype: _type_
    """
    # cols and rows definitions
    cols = plots if plots <= max_col else max_col
    rows = int(plots / max_col) + int(plots % max_col != 0)

    fig, axes = plt.subplots(rows,
                             cols,
                             figsize=(cols * fig_frame[0], rows * fig_frame[1]),
                             dpi=res)
    if plots > 1:
        axes = axes.flatten()
        for i in range(plots, max_col*rows):
            remove_frame(axes[i])
    elif plots == 1:
        pass
    
    return fig, axes


def remove_frame(axes) -> None:
    for side in ['bottom', 'right', 'top', 'left']:
        axes.spines[side].set_visible(False)
    axes.set_yticks([])
    axes.set_xticks([])
    axes.xaxis.set_ticks_position('none')
    axes.yaxis.set_ticks_position('none')
    pass

# --- ////////////// ---#