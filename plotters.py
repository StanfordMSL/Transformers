import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from typing import List

def tXU_to_3D(tXU_list:List[np.ndarray],
              n:int=500,plot_last:bool=False):

    # Initialize World Frame Plot
    traj_colors = ["red","green","blue","orange","purple","brown","pink","gray","olive","cyan"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plim = np.array([
        [ -8.0,  8.0],
        [ -3.0,  3.0],
        [  1.0, -3.0]])
    
    xlim = plim[0,:]
    ylim = plim[1,:]
    zlim = plim[2,:]
    ratio = plim[:,1]-plim[:,0]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_box_aspect(ratio)  # aspect ratio is 1:1:1 in data space

    ax.invert_zaxis()

    # Rollout the world frame trajectory
    for idx,tXU in enumerate(tXU_list):
        # Plot the world frame trajectory
        ax.plot(tXU[1,:], tXU[2,:], tXU[3,:],color=traj_colors[idx%len(traj_colors)],alpha=0.5)             # spline

        for i in range(0,tXU.shape[1],n):
            quad_frame(tXU[1:14,i],ax)

        if plot_last == True or idx == 0:
            quad_frame(tXU[1:14,-1],ax)

    plt.show()

def quad_frame(x:np.ndarray,ax:plt.Axes):
    frame_body = np.diag([0.6,0.6,-0.2])
    frame_labels = ["red","green","blue"]
    pos  = x[0:3]
    quat = x[6:10]
    
    for j in range(0,3):
        Rj = R.from_quat(quat).as_matrix()
        arm = Rj@frame_body[j,:]

        frame = np.zeros((3,2))
        if (j == 2):
            frame[:,0] = pos
        else:
            frame[:,0] = pos - arm

        frame[:,1] = pos + arm

        ax.plot(frame[0,:],frame[1,:],frame[2,:], frame_labels[j],label='_nolegend_')
