#!/usr/bin/env python
# coding: utf-8

# # 1800*600
# 
# 
# - test parell
# - test if need weak seed 
# - test other viscosity

# In[1]:


from underworld import UWGeodynamics as GEO

import underworld as uw
from underworld import function as fn
import numpy as np
import math

u = GEO.UnitRegistry
#GEO.__version__


# In[2]:


# In[3]:


velocity = 2.5 * u.centimetre / u.year

# Scaling
T0 = 293.15 * u.degK # 20 * u.degC
Tz = 1733.15 * u.degK # 1500 * u.degC

K_viscosity = 1e21  * u.pascal * u.second
K_density = 3300 * u.kilogram / (u.meter)**3
KH = 600 * u.kilometer
K_gravity =  10. * u.meter / u.second**2

Kt = KH/velocity
bodyforce = K_density  * K_gravity 
KM = bodyforce * KH**2 * Kt**2
KT = Tz - T0

GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[length]"] = KH
GEO.scaling_coefficients["[mass]"] = KM
#GEO.scaling_coefficients["[temperature]"] = KT
    
GEO.rcParams["swarm.particles.per.cell.2D"]= 20    


# In[4]:


nx0 = 720   #720
ny0 = 240  #240

minCoord0 = (0. * u.kilometer, -600. * u.kilometer)
maxCoord0 = (1800 * u.kilometer,0. * u.kilometer )

# minCoord1 = (0. * u.kilometer, -300. * u.kilometer)
# maxCoord1 = (1200. * u.kilometer,0. * u.kilometer )

Model = GEO.Model(elementRes=(nx0,ny0),
                  minCoord=minCoord0,  
                  maxCoord=maxCoord0,
                  gravity=(0.0, -K_gravity))

Model.outputDir= "op_M0"
Model.minStrainRate = 1e-18 / u.second


# In[5]:


Z_oc = -35. * u.kilometer
Z_ic = -35. * u.kilometer
Z_ac = -35. * u.kilometer

Z_oml = -150. * u.kilometer
Z_iml = -150. * u.kilometer
Z_aml = -150. * u.kilometer

Z_ma = -600. * u.kilometer


X0_oc = 0. * u.kilometer
X1_oc = 300. * u.kilometer
X2_oc = 1500. * u.kilometer
X3_oc = 1800. * u.kilometer 

X0_ic = 300. * u.kilometer
X1_ic = 600. * u.kilometer
X2_ic = 1200. * u.kilometer
X3_ic = 1500. * u.kilometer 

X0_ac = 600. * u.kilometer
X1_ac = 1200. * u.kilometer

# h1 = 0.5*u.kilometer
# h2 = -2.5*u.kilometer

h1 = 0.*u.kilometer
h2 = 0.*u.kilometer

w_trans1 = 50. * u.kilometer
w_trans2 = 0. * u.kilometer
w_trans3 = 0. * u.kilometer

topoy1 =  GEO.nd(h1) #fn.misc.constant([-
topoy2 =  GEO.nd(h2)
topox1 =  GEO.nd(X1_ic)

meshz0 = GEO.nd(maxCoord0[1])
meshz1 = GEO.nd(minCoord0[1])
meshz2 = GEO.nd(minCoord0[1])


dx = GEO.nd(maxCoord0[0]/nx0)
dy = -GEO.nd(minCoord0[1]/ny0)

# dx = GEO.nd(maxCoord0[0]/nx0)
# dy1 =  GEO.nd(3.*u.kilometer)
# dy2 = GEO.nd(30.*u.kilometer)

# ny1 = int(np.round((meshz1-meshz2)/dy2))
# ny2 = int(ny0-ny1) 

# Model.fsmeshny = ny2
# Model.fsmeshnx = nx0
# Model.fsmeshny1 = ny1

# Model.fsminCoord = tuple([GEO.nd(val) for val in minCoord1])
# Model.fsmaxCoord = tuple([GEO.nd(val) for val in maxCoord1])


# In[6]:


def find_IndexSet(axis):
    Sets = Model.mesh.specialSets["Empty"]
    for index in axis:
        Sets.add(index)
    return Sets


# In[7]:


# IndexSets for compensation depth
axis_cd = np.where((Model.mesh.data[:,1]<=meshz1+dy/4)&(Model.mesh.data[:,1]>=meshz1-dy/4))
Sets_cd = Model.mesh.specialSets["Empty"]
for index in axis_cd:
    Sets_cd.add(index)
    
meshz_check = GEO.nd(-150*u.kilometer)
axis_check = np.where((Model.mesh.data[:,1]<=meshz_check+dy/4)&(Model.mesh.data[:,1]>=meshz_check-dy/4))
Sets_check = Model.mesh.specialSets["Empty"]
for index in axis_check:
    Sets_check.add(index)    
    
# meshz_check2 = GEO.nd(-300*u.kilometer)
# axis_check2 = np.where((Model.mesh.data[:,1]<=meshz_check2+dy/4)&(Model.mesh.data[:,1]>=meshz_check2-dy/4))
# Sets_check2 = Model.mesh.specialSets["Empty"]
# for index in axis_check2:
#     Sets_check2.add(index)    
    
# meshz_check1 = GEO.nd(-10*u.kilometer)
# axis_check1 = np.where((Model.mesh.data[:,1]<=meshz_check1+dy/4)&(Model.mesh.data[:,1]>=meshz_check1-dy/4))
# Sets_check1 = Model.mesh.specialSets["Empty"]
# for index in axis_check1:
#     Sets_check1.add(index)      


# In[8]:


oc_Shape1 = GEO.shapes.Polygon([(X0_oc,  h2),
                               (X1_oc,h2),
                               (X1_oc,  Z_oc),
                               (X0_oc, Z_oc)])

oc_Shape2 = GEO.shapes.Polygon([(X2_oc,  h2),
                               (X3_oc,h2),
                               (X3_oc,  Z_oc),
                               (X2_oc, Z_oc)])
oc_Shape = oc_Shape2 | oc_Shape1
    
oml_Shape1 = GEO.shapes.Polygon([(X0_oc, Z_oc),
                                (X1_oc, Z_oc),
                               (X1_oc, Z_oml),
                               (X0_oc,Z_oml)])

oml_Shape2 = GEO.shapes.Polygon([(X2_oc, Z_oc),
                                (X3_oc, Z_oc),
                               (X3_oc, Z_oml),
                               (X2_oc,Z_oml)])
oml_Shape = oml_Shape2 | oml_Shape1


ic_Shape1 = GEO.shapes.Polygon([(X0_ic,  h1),
                               (X1_ic,  h1),
                               (X1_ic, Z_ic),
                               (X0_ic, Z_ic)])
ic_Shape2 = GEO.shapes.Polygon([(X2_ic,  h1),
                               (X3_ic,  h1),
                               (X3_ic, Z_ic),
                               (X2_ic, Z_ic)])
ic_Shape = ic_Shape2 | ic_Shape1

iml_Shape1 = GEO.shapes.Polygon([(X0_ic,  Z_ic),
                               (X1_ic,  Z_ic),
                               (X1_ic, Z_iml),
                               (X0_ic, Z_iml)])

iml_Shape2 = GEO.shapes.Polygon([(X2_ic,  Z_ic),
                               (X3_ic,  Z_ic),
                               (X3_ic, Z_iml),
                               (X2_ic, Z_iml)])
iml_Shape = iml_Shape2 | iml_Shape1
ac_Shape = GEO.shapes.Polygon([(X0_ac,  h1),
                               (X1_ac,  h1),
                               (X1_ac, Z_ac),
                               (X0_ac, Z_ac)])

aml_Shape = GEO.shapes.Polygon([(X0_ac,  Z_ac),
                               (X1_ac,  Z_ac),
                               (X1_ac, Z_aml),
                               (X0_ac, Z_aml)])

ma_Shape = GEO.shapes.Layer2D(top=0. * u.kilometer, bottom=Z_ma)


# In[9]:


ma   = Model.add_material(name="Mantle Asthenosphere", shape=ma_Shape)

oc = Model.add_material(name="Oceanic Crust", shape=oc_Shape)
oml  = Model.add_material(name="Oceanic Mantle Lithosphere", shape=oml_Shape)

ac   = Model.add_material(name="Continental Crust", shape=ac_Shape)
aml  = Model.add_material(name="Continental Mantle Lithosphere", shape=aml_Shape)

ic    = Model.add_material(name="Indentor Crust", shape=ic_Shape)
iml  = Model.add_material(name="Indentor Mantle Lithosphere", shape=iml_Shape)


# In[10]:


npoints = 250
coords0 = np.ndarray((npoints, 2))
x = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints)
y = (x-GEO.nd(X1_ic))/GEO.nd(w_trans1)*(GEO.nd(Z_oc)-GEO.nd(Z_ic))+GEO.nd(Z_ic)
y[x<=GEO.nd(X0_oc)] = GEO.nd(Z_ic)
y[x>=GEO.nd(X0_oc)+GEO.nd(w_trans1)] = GEO.nd(Z_oc)
coords0[:, 0] = x
coords0[:, 1] = y


coords1 = np.ndarray((npoints, 2))
x = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints)
y = (x-GEO.nd(X1_ic))/GEO.nd(w_trans1)*(GEO.nd(Z_oml)-GEO.nd(Z_iml))+GEO.nd(Z_iml)
y[x<=GEO.nd(X0_oc)] = GEO.nd(Z_iml)
y[x>=GEO.nd(X0_oc)+GEO.nd(w_trans1)] = GEO.nd(Z_oml)
coords1[:, 0] = x
coords1[:, 1] = y

Model.add_passive_tracers(name="Moho", vertices=coords0)
Model.add_passive_tracers(name="LAB", vertices=coords1)


# In[11]:




# In[12]:


ma.density = 3250 * u.kilogram / u.metre**3
#water.density = 1000 * u.kilogram / u.metre**3

oc.density = 2800 * u.kilogram / u.metre**3
oml.density = 3300 * u.kilogram / u.metre**3

ac.density = 2800 * u.kilogram / u.metre**3
aml.density = 3300 * u.kilogram / u.metre**3

ic.density = 2800 * u.kilogram / u.metre**3
iml.density = 3300 * u.kilogram / u.metre**3


# In[13]:


Model.minViscosity = 1e20 * u.pascal * u.second
Model.maxViscosity = 1e23 * u.pascal * u.second
ma.viscosity = 1e20 * u.pascal * u.second
oc.viscosity = 1e23 * u.pascal * u.second
oml.viscosity = 1e21 * u.pascal * u.second

ac.viscosity = 1e23 * u.pascal * u.second
aml.viscosity = 1e21 * u.pascal * u.second

ic.viscosity = 1e23 * u.pascal * u.second
iml.viscosity = 1e21 * u.pascal * u.second


# In[14]:


axis_ic = np.where((Model.mesh.data[:,0]<=GEO.nd(X1_ic)+dx/4)&(Model.mesh.data[:,0]>=GEO.nd(X0_ic)-dx/4)&(Model.mesh.data[:,1]>=GEO.nd(Z_iml)))
IndexSet_icL = find_IndexSet(axis_ic)

axis_ic = np.where((Model.mesh.data[:,0]<=GEO.nd(X3_ic)+dx/4)&(Model.mesh.data[:,0]>=GEO.nd(X2_ic)-dx/4)&(Model.mesh.data[:,1]>=GEO.nd(Z_iml)))
IndexSet_icR = find_IndexSet(axis_ic)


# In[15]:


Model.set_velocityBCs(left=[0., None], right=[0., None], top=[None, None], bottom=[0.,0.],nodeSets=[(IndexSet_icL,[velocity,0.]),(IndexSet_icR,[-velocity,0.])])


# In[16]:


# Model.set_temperatureBCs(top=T0, 
#                          bottom=Tz)


# In[17]:


#Model._temperatureDot = None
#Model._temperature = None


# In[18]:


#Model.init_model(pressure="lithostatic",temperature=None)


# In[19]:



# In[21]:

Model.freeSurface = True


# In[22]:

checkpoint_interval = 1e2*u.kiloyear
dt = 1.*u.kiloyear
Model.run_for(5.0 * u.megayears, checkpoint_interval=checkpoint_interval,dt= dt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




