# -------------------------------------------------- #
# Plot utilities
# for the Active Learning Classifier
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.preprocessing import MinMaxScaler

from typing import Tuple, Callable, List

from .misc import get_space_lims, make_meshgrid
from activeclf.acquisition import sampling_fps


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

    print('classes:', n_classes)
    CMAPS = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']

    # Build the feature space PDF and Entropy
    Z = clfModel.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = MinMaxScaler().fit_transform(X=Z)
    H = np.round(scipy.stats.entropy(pk=Z, axis=1),2)
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


def plot_classification2D(data: np.ndarray,
                          feature_variable: List[str], 
                          clfModel: Callable,
                          points_ndx: List[int]=None) -> None:
    
    # Buildi the complete feature space (visualization)
    X0, X1 = data[feature_variable[0]], data[feature_variable[1]]
    y = clfModel.clf.predict(data)
    incr = '5%'
    x_min, x_max = get_space_lims(coord=X0, incr=incr)
    y_min, y_max = get_space_lims(coord=X1, incr=incr)
    xx, yy = make_meshgrid(x=X0, y=X1, incr=incr, delta=.1)

    # Classes specification and color grading
    n_classes = len(np.unique(y))
    CMAPS = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']

    # Build the feature space PDF and Entropy
    Z = clfModel.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    H = scipy.stats.entropy(pk=Z, axis=1)
    Z = Z.reshape((xx.shape[0], xx.shape[1], -1))

    # plots
    fig, ax = get_axes(n_classes+2,3)

    for i in range(n_classes):
        # plot the initial data
        alphas = get_alphas(Z=Z[:,:,i])
        ax[i].scatter(X0, X1, c='0.', marker='.', s=25, alpha=.5, zorder=1)
        if points_ndx:
            ax[i].scatter(X0.iloc[points_ndx], X1.iloc[points_ndx],
                          c=y[points_ndx], edgecolor='0.',
                          marker='o', s=50, alpha=1., zorder=1)
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

    # ax[n_classes].scatter(X0, X1, c='0.', marker='.', s=25, zorder=4)
    ax[n_classes].scatter(X0, X1, c='0.', marker='.', s=25, alpha=.5, zorder=1)
    if points_ndx:
        ax[n_classes].scatter(X0.iloc[points_ndx], X1.iloc[points_ndx],
                              c=y[points_ndx], edgecolor='0.',
                              marker='o', s=50, alpha=1., zorder=1)
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

    ax[n_classes+1].scatter(X0, X1, c='0.', marker='.', s=25, alpha=.5, zorder=1)
    if points_ndx:
        ax[n_classes+1].scatter(X0.iloc[points_ndx], X1.iloc[points_ndx],
                                c=y[points_ndx], edgecolor='0.',
                                marker='o', s=50, alpha=1., zorder=1)
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
    
    CMAPS = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
    
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


def plot_entropy3D(X, Z, decimals, scaling=True, orientation=(-145,12)):
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

    ax.azim, ax.elev = orientation
    ax.dist = 10

    fig.tight_layout()


def plot_entropy(X, Z, decimals=2, levels=5, scaling=True):
    # get the space
    X0, X1 = X

    # compute H
    if scaling:
        Z = MinMaxScaler().fit_transform(X=Z)
    H = np.around(scipy.stats.entropy(pk=Z, axis=1), decimals=decimals)
    H_val_chunks = np.sort(np.unique(H))[::-1][:levels]

    palette = sns.color_palette(palette='colorblind', n_colors=levels)

    H_idx_chunks_list = [
        [id for id,val in enumerate(H) if val == hchunk]
        for hchunk in H_val_chunks
        ]
    
    fig, ax = get_axes(2,2)
    ax[0].scatter(X0, X1, c=H, cmap='plasma', s=H*70, marker='o', zorder=1)
    ax[0].set_xlabel('f0')
    ax[0].set_ylabel('f1')
    ax[0].set_title('H space')
    ax[0].set_xticks(())
    ax[0].set_yticks(())

    ax[1].scatter(X0, X1, facecolor='none', edgecolor='0.', s=H*70, marker='o', zorder=2)
    ax[1].set_xlabel('f0')
    ax[1].set_ylabel('f1')
    ax[1].set_title('discretization')
    ax[1].set_xticks(())
    ax[1].set_yticks(())
    for i,idx in enumerate(H_idx_chunks_list):
        ax[1].scatter(X0[idx], X1[idx], c=[palette[i]]*len(idx), s=H[idx]*40, 
                      label=f'{H_val_chunks[i]} ({len(idx)})',
                      marker='o', zorder=1)
        
    ax[1].legend(title='H Levels', loc='center left', bbox_to_anchor=(1, 0.5))

    pass


def plot_entropy_selection(X, Z, num_points, levels=5, decimals=2, scaling=True):

    # get the space
    XX, X0, X1 = X

    # compute H
    if scaling:
        Z = MinMaxScaler().fit_transform(X=Z)
    H = np.around(scipy.stats.entropy(pk=Z, axis=1), decimals=decimals)
    H_max_ndx = np.argmax(H)
    H_val_chunks = np.sort(np.unique(H))[::-1][:levels]

    H_idx_chunks_list = [
        [id for id,val in enumerate(H) if val == hchunk]
        for hchunk in H_val_chunks
        ]
    
    subspace = get_subspace(indexes=H_idx_chunks_list, points=num_points)
    palette = sns.color_palette(palette='colorblind', n_colors=levels)

    fig, ax = get_axes(3, 2)
    ax[0].scatter(X0, X1, c=H, cmap='plasma', s=H*70, marker='o', zorder=1)
    ax[0].set_xlabel('f0')
    ax[0].set_ylabel('f1')
    ax[0].set_xticks(())
    ax[0].set_yticks(())
    for i,sub in enumerate(subspace):
        ax[0].scatter(X0[sub], X1[sub], c=[palette[i]]*len(sub), s=H[sub]*35, marker='*', zorder=2)

    ax[1].scatter(X0, X1, facecolor='none', edgecolor='0.', s=H*70, marker='o', zorder=2)
    ax[1].set_xlabel('f0')
    ax[1].set_ylabel('f1')
    ax[1].set_xticks(())
    ax[1].set_yticks(())
    for i,idx in enumerate(H_idx_chunks_list):
        ax[1].scatter(X0[idx], X1[idx], c=[palette[i]]*len(idx), s=H[idx]*40, 
                      label=f'{H_val_chunks[i]}',
                      marker='o', zorder=1)

    ax[1].legend(title='H Levels', loc='center left', bbox_to_anchor=(1, 0.5))

    _fps_points = sampling_fps(X=XX.iloc[np.concatenate(subspace)],
                               n=num_points, start_idx=0)
    fps_points = [np.concatenate(subspace)[sp] for sp in _fps_points]
    ax[2].scatter(X0, X1, c=H, cmap='plasma', s=H*70, marker='o', alpha=.2, zorder=1)
    ax[2].set_xlabel('f0')
    ax[2].set_ylabel('f1')
    ax[2].set_xticks(())
    ax[2].set_yticks(())
    for i,sub in enumerate(subspace):
        ax[2].scatter(X0[sub], X1[sub], c=[palette[i]]*len(sub), s=H[sub]*35, alpha=.5, marker='*', zorder=2)
    ax[2].scatter(X0[fps_points[0]], X1[fps_points[0]], 
                  c='r', label='Start', edgecolor='0.',
                  s=50, zorder=3)
    ax[2].scatter(X0[fps_points], X1[fps_points], 
                  facecolor='none', edgecolor='0.',
                  s=50, zorder=3)
    ax[2].scatter(X0[fps_points[-1]], X1[fps_points[-1]], 
                  c='b', label='End', edgecolor='0.',
                  s=50, zorder=3)
    points = np.array([X0[fps_points], X1[fps_points]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('vlag_r'), norm=plt.Normalize(0, 10))
    lc.set_array(np.arange(len(fps_points)))
    lc.set_linewidth(2)
    lc.autoscale()
    ax[2].add_collection(lc)
    ax[2].legend(title='FPS Path', loc='lower right')

    # fig.tight_layout()
    pass


def get_subspace(indexes: list[list[int]], points: int) -> list[list[int]]:
    sub_space = list()
    
    while True:

        for subidx in indexes:
            sub_space.append(subidx)

            if len(np.concatenate(sub_space)) >= points:
                break
    
        return sub_space
    

def plot_arrow_path(x,y, ax):
    u = np.diff(x)
    v = np.diff(y)
    pos_x = x[:-1] + u/2
    pos_y = y[:-1] + v/2
    norm = np.sqrt(u**2+v**2)
    ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")


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