#!/usr/bin/env python
# -*- coding: utf-8 -*

"""deepsensemaking (dsm) ml sub-module"""

import numpy as np

def lincluster3d(xmin,xmax,ymin,ymax,zmin,zmax,nsamp=100,xvar=2,yvar=2,zvar=2,):
    """
    Example usage:

      fig,ax = plt.subplots(1,1,figsize=(6,6,),subplot_kw=dict(projection='3d'))

      x0,y0,z0 = lincluster3d(121,240,121,180,121,240,120,5,5,5,)
      x1,y1,z1 = lincluster3d(201,300,180,120,201,460,120,5,5,5,)
      x2,y2,z2 = lincluster3d(221,340,180,221,120,340,120,5,5,5,)

      ax.scatter(x0,y0,z0,color="C0",linewidth=0.5,label="D0",)
      ax.scatter(x1,y1,z1,color="C1",linewidth=0.5,label="D1",)
      ax.scatter(x2,y2,z2,color="C2",linewidth=0.5,label="D2",)

      # ax.scatter(x0,y0,z0,c=z,cmap='viridis',linewidth=0.5);

      ax.set_xlabel("$X$",fontsize=12,rotation=0,)
      ax.set_ylabel("$Y$",fontsize=12,rotation=0,)
      ax.set_zlabel("$Z$",fontsize=12,rotation=0,)

      ax.set_xlim(0,500,)
      ax.set_ylim(0,500,)
      ax.set_zlim(0,500,)
      ax.legend()
      # ax.view_init(azim=-120,elev=30,)
      ax.view_init(azim=-92,elev=20,)
      # ax.set_aspect("equal")
      ax.set_aspect("auto")

    """
    nsamp *= 1j
    x0 = np.mgrid[xmin:xmax:nsamp]
    y0 = np.mgrid[ymin:ymax:nsamp]
    z0 = np.mgrid[zmin:zmax:nsamp]
    x0 += np.random.normal(loc=0,scale=xvar,size=x0.shape)
    y0 += np.random.normal(loc=0,scale=yvar,size=y0.shape)
    z0 += np.random.normal(loc=0,scale=zvar,size=z0.shape)
    return x0,y0,z0


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    See
      https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range  = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range  = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range  = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle-plot_radius,x_middle+plot_radius])
    ax.set_ylim3d([y_middle-plot_radius,y_middle+plot_radius])
    ax.set_zlim3d([z_middle-plot_radius,z_middle+plot_radius])
