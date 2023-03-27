#!/usr/bin/env python
# coding: utf-8

# Rayleigh-Taylor instability
# ======
# 
# This notebook models the Rayleigh-Taylor instability outlined in Kaus *et al.* (2010). 
# 
# **Keywords:** Stress state, Free surface
# 
# **References**
# 1. Kaus, B. J., Mühlhaus, H., & May, D. A. (2010). A stabilization algorithm for geodynamic numerical simulations with a free surface. Physics of the Earth and Planetary Interiors, 181(1-2), 12-20.
# 
# ![](./images/kaus2010RTI.gif)

# In[1]:


from underworld import UWGeodynamics as GEO
#from underworld import visualisation as vis

import underworld.function as fn
import math
import numpy as np


# In[2]:


u = GEO.UnitRegistry

KL = 500 * u.kilometer
K_viscosity = 1e20  * u.pascal * u.second
K_density   = 3200 * u.kilogram / u.meter**3

KM = K_density * KL**3
Kt = KM/ ( KL * K_viscosity )

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM


# In[3]:


Model = GEO.Model(elementRes=(50,50),
                  minCoord=(-250. * u.kilometer, -500. * u.kilometer),  
                  maxCoord=(250. * u.kilometer, 0. * u.kilometer),
                  gravity=(0.0, -9.81 * u.meter / u.second**2))

dt = 2.5*u.kiloyear
dt_str = "%.1f" %(dt.m)
checkpoint_interval = 1e2*u.kiloyear
fdir = "1_23_02_FreeSurface_Kaus2010_Rayleigh-Taylor_Instability_dt"+dt_str+"ka"
Model.outputDir = fdir


# In[4]:


wavelength = GEO.nd(KL)
amplitude  = GEO.nd(5*u.kilometer)
offset     = GEO.nd(-100.*u.kilometer)
k = 2. * math.pi / wavelength

coord = fn.coord()
perturbationFn = offset + amplitude*fn.math.cos(k*coord[0])

lightShape = perturbationFn > coord[1] 
densShape  = perturbationFn <= coord[1]

densMaterial = Model.add_material(name="Dense Material", shape=densShape)
lightMaterial = Model.add_material(name="Light Material", shape=lightShape)


# In[5]:


densMaterial.density  = 3300 * u.kilogram / u.metre**3
lightMaterial.density = 3200 * u.kilogram / u.metre**3

densMaterial.viscosity = 1e21 * u.pascal * u.second
lightMaterial.viscosity = 1e20 * u.pascal * u.second


# In[7]:


npoints = 1000
coords = np.ndarray((npoints, 2))
coords[:, 0] = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints)
coords[:, 1] = offset + amplitude*np.cos(k*coords[:, 0])

Model.add_passive_tracers(name="interface", vertices=coords)


# In[8]:


# Fig = vis.Figure(figsize=(500, 500))
# Fig.Points(Model.interface_tracers, pointSize=5.0)
# Fig.Points(Model.swarm, Model.materialField,pointSize=3.,colourBar=False)
# Fig.Mesh(Model.mesh)
# Fig.save("Fig_Kaus2010RTI_0.png")
# Fig.show()


# In[9]:


Model.set_velocityBCs(left=[0., None], right=[0., None], top=[None, None], bottom=[0.,0.])
Model.freeSurface = True


# In[10]:


Model.run_for(5.5 * u.megayears, checkpoint_interval=checkpoint_interval,dt= dt)


# In[1]:


# Fig.save("Fig_Kaus2010RTI_1.png")
# Fig.show()


# In[ ]:





# In[2]:


# import h5py   
# import matplotlib.pyplot as plt
# import numpy as np
# import math

# def load_surf_swarm(fdir,step):
#     fname = fdir+"interface-"+str(step)+".h5"
#     fh5   = h5py.File(fname ,'r')  
#     fdata = fh5["data"][()]
#     xcoord = fdata[:,0]
#     ycoord = fdata[:,1]
#     return xcoord,ycoord

# def load_depth(fdir,maxstep,dstep):
#     depth_l = []
#     depth_r = []
#     for step in range(0,maxstep+1,dstep):
#         xcoord,ycoord = load_surf_swarm(fdir,step)
#         depth_l.append(ycoord[0])   
#         depth_r.append(ycoord[-1])  
#     return np.array(depth_l),np.array(depth_r)


# In[18]:


# dt0,maxsteps0,dstep0,= 100,54,1
# times0 = np.arange(0,dt0*maxsteps0+dt0*dstep0/2,dt0*dstep0)

# fdir += "/"
# depth0_l,depth0_r = load_depth(fdir,maxsteps0,dstep0)


# In[19]:


# # Fig 3 in Kaus et al., 2010

# fname = "Depth of the interface at x=−250km versus time for the free surface simulations"
# fig, ax1 = plt.subplots(nrows=1, figsize=(8,6))
# ax1.set(xlabel='Time [Myrs]', ylabel='Interface Depth [km]') 
# ax1.plot(times0/1000,depth0_l,'-k')
# ax1.set_ylim([-500,-100])
# ax1.set_xlim([0,6])
# ax1.grid()
# #ax1.legend(loc = 'lower right',prop = {'size':8})
# plt.savefig(fname,dpi=150,bbox_inches='tight')


# In[ ]:




