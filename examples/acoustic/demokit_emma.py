# Add module path
import sys
import os

from scipy import ndimage
import numpy as np
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from matplotlib.patches import Circle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage


sys.path.append('/Users/emmapearce/projects/opesci/devito/')
sys.path.append('/Users/emmapearce/projects/opesci/examples/')
sys.path.append('/Users/emmapearce/projects/opesci/acoustic/')

from examples.containers import IShot, IGrid
from examples.acoustic.Acoustic_codegen import Acoustic_cg
from devito import clear_cache

# Define geometry
model = None
dx = 50 
lx = 3000 
ly = 10000
sd = 300 #sea depth in m 
origin = (0,0)
dimensions =(int(lx/dx), int(ly/dx))

nsrc = 101 #what is nsrc 
spc_order = 4 #what is spacial order. 

# Read velocity    
    
def get_true_model(lx,ly,dx,water_depth):
    global problem_spec
    size = (int(lx/(dx)), int(ly/(dx)))
    model_true = np.ones(size) 
    sf_grid_depth = water_depth/dx  # puts depth of sea floor into grid spacing defined by dx
    max_v= 3000. # m/s  velocity at bottom of sea bed 
    seabed_v = 1700. # m/s velocity at top of seabed 
    water_v = 1500. #m/s water velocity 
    m = (max_v-seabed_v)/(size[0]-1-sf_grid_depth)  # velocity gradient of seabed THIS IS WEIRD 
    
    #SET VELOCITY OF SEABED (uses velocity gradient m)
    for i in range (sf_grid_depth, size[0]):
        model_true[i][:] = (m*(i-sf_grid_depth)) + seabed_v
            
    #ADD CIRCLE ANOMALIES - REFRACTORS   
    radius = 500./dx  #radius of both circles
    cx1, cy1 = 7500./dx, 1250./dx # The center of circle  
    yc1, xc1 = np.ogrid[-radius: radius, -radius: radius]
    index = xc1**2 + yc1**2 <= radius**2
    model_true[cy1-radius:cy1+radius, cx1-radius:cx1+radius][index] =  2900.   #positive velocity anomaly 
    
    cx2, cy2 = 2500./dx, 1250./dx # The center of circle 2 
    yc2, xc2 = np.ogrid[-radius: radius, -radius: radius]
    index = xc2**2 + yc2**2 <= radius**2
    model_true[cy2-radius:cy2+radius, cx2-radius:cx2+radius][index] = 1700.   #negative velocity anomaly 
    #plot = plt.imshow(model_true)
    
    #BLURR CIRCLES
    blurred_model = gaussian_filter(model_true,sigma=2)   

    #ADD REFLECTORS - NEGATIVE ANOMS     
    ex1, ey1 =3000./dx, 2250./dx # The center of reflector 1   
    rx, ry =350./dx,  75./dx
    
    ye, xe = np.ogrid[-radius: radius, -radius: radius]
    index = (xe**2/rx**2) + (ye**2/ry**2) <= 1 
    blurred_model[ey1-radius:ey1+radius, ex1-radius:ex1+radius][index] = 2000.   
    
    ex2, ey2 =7250./dx, 2150./dx # The center of reflector 2 
    rx, ry =200./dx, 75./dx     #up and down radius 
    
    ye2, xe2 = np.ogrid[-radius: radius, -radius: radius]
    index = (xe2**2/rx**2) + (ye2**2/ry**2) <= 1 
    blurred_model[ey2-radius:ey2+radius, ex2-radius:ex2+radius][index] = 2000.  
    
    #SET VELOCITY OF WATER 
    for i in range(0,sf_grid_depth,1):
        blurred_model[i][:] = 1500.
     
    true_model = blurred_model 
    vp = true_model 

    # Do i need 2d slice of 3d model? 
    # Create exact model
    global model
    model = IGrid()
    model.create_model([origin[0],origin[1]],[dx,dx], vp)  #spacing put as dx? 
    
    trueplot = plt.imshow(true_model) 
    plt.savefig("true_model.png")
    
    return model 

get_true_model(lx,ly,dx,sd)

###################################      

def get_initial_model(lx,ly,dx,water_depth):
    global model
    if model is None:
        model = get_true_model()

    # Smooth velocity
    size = (int(lx/(dx)), int(ly/(dx)))   # array can only be whole numbers took away the +1 
    initial_model0 = np.ones(size) 
    sf_grid_depth = water_depth/dx  # puts depth of sea floor into grid spacing defined by dx
    max_v= 3000. # m/s  velocity at bottom of sea bed 
    seabed_v = 1700. # m/s velocity at top of seabed 
    water_v = 1500. #m/s water velocity 
    m = (max_v-seabed_v)/(size[0]-1-sf_grid_depth)  # velocity gradient of seabed  
    
    for i in range (sf_grid_depth, size[0]):
        initial_model0[i][:] = (m*(i-sf_grid_depth)) + seabed_v
    
    for i in range(sf_grid_depth):
        initial_model0[i][:] = 1500.
    
    smooth_vp = initial_model0 
    #global problem_spec
    model0 = IGrid()
    model0.create_model([origin[0],origin[1]], [dx,dx], smooth_vp)
    
    initialplot = plt.imshow(initial_model0) 
    plt.savefig("initial_model.png")
    
    return model0
    
get_initial_model(lx,ly,dx,sd)    


######
# Source function: Set up the source as Ricker wavelet for f0
def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))
    return (1-2.*r**2)*np.exp(-r**2)  #defines the ricker wave 

def get_shot(i):
    global model
    if model is None:
        model = get_true_model()

    # Define seismic data.
    data = IShot()

    f0 = .015     
    dt = model.get_critical_dt()
    t0 = 0.0
    tn = 1500    
    nt = int(1+(tn-t0)/dt)

    time_series = source(np.linspace(t0, tn, nt), f0)
    
    nsrc =  nsrc
    spacing = [dx,dx]
    origin = [origin[0], origin[1]]
    dimensions = [dimensions[0],dimensions[1]][1:]
    
    print "i am here"
    
    receiver_coords = np.zeros((nsrc, 2))
    receiver_coords[:, 0] = np.linspace(2 * dx,
                                        origin[0] + (dimensions[0] - 2) * dx,
                                        num=nsrc)
    receiver_coords[:, 1] = origin[1] + 2 * dx
    data.set_receiver_pos(receiver_coords)
    data.set_shape(nt, nsrc)
        
    sources = np.linspace(2 * dx, origin[0] + (dimensions[0] - 2) * dx,num=nsrc)

    location = (sources[i], origin[1] + 2 * dx)
    data.set_source(time_series, dt, location)

    Acoustic = Acoustic_cg(model, data, t_order=2, s_order=4)
    rec, u, gflopss, oi, timings = Acoustic.Forward(save=False, cse=True)
    
    ricker = plt.plot(time_series)
    plt.savefig("rickerwave.png")
    
    return data, rec 

