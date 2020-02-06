###### To implement changes in fastjet: 
#-- (pip install cython on xenia)
#-- Add functionality from fastjet to fastjet.pxd file and __libpyjet.pyx
#-- in pyjet/src: cython --cplus _libpyjet.pyx (this generates .cpp file)
#-- in pyjet: make  (or python3 setup.py build_ext --inplace)

##########TODO 
#Aplanarity: implement SphericityTensor/CenterOfMassTool
##########DONE:
#Split12, Split23: KTsplitting tool
#ZCut12 (needs KTSplitting tool)
#Qw
#C2,D2 (ECF = energy correlation functions) 
#Tau1, tau2, tau3 (Tau1_wta), Tau21, Tau23, Tau13 
#PlanarFlow
#Angularity
#KtDR


import h5py    
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from skhep.math.vectors import *
#from scipy.special import softmax
import pickle
import glob
from pprint import pprint

import sys 
sys.path.append("/Users/juliagonski/Documents/Columbia/Physics/yXH/test_pyjet_extfastjet/pyjet")
from pyjet import DTYPE_PTEPM,ClusterSequence,JetDefinition,PseudoJet,cluster,EnergyCorrelator,EnergyCorrelatorC2,EnergyCorrelatorD2,Nsubjettiness,NsubjettinessRatio,KT_Axes,NormalizedMeasure,LundGenerator


#-------------------------------------------------------------------
def calc_lund_planes(jet):
  print('Jet: ', jet)
  LundGen = LundGenerator('cambridge')
  print('From dedicated tool: ', LundGen.result(jet))
  return LundGen.result(jet) 



##########################################################################
#-------------------------------------------------------------------------
if __name__ == "__main__":

  #print(dir(pyjet))

  f = pd.read_hdf("/Users/juliagonski/Documents/Columbia/Physics/yXH/lhcOlympics2020/events_anomalydetection.h5", start=0, stop=100)
  events_combined = f.T
  np.shape(events_combined)

  n_consts = 11
  n_start = 10


  #Now, let's cluster some jets!
  sorted_lund_planes = {}
  for i in range(n_start,n_consts):
    sorted_lund_planes.update({str(i):[]})

  for n_c in range(n_start, n_consts):
    print('Current n_Consts: ', n_c)

    #for mytype in ['background','signal']:
    for i in range(len(events_combined)):
      if (i%10==0):
          print(i)
          pass
      issignal = events_combined[i][2100]
      pseudojets_input = np.zeros(len([x for x in events_combined[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
      for j in range(700): #number of hadrons per event
          if (events_combined[i][j*3]>1.0):   #min pT cut, could make this tighter
              pseudojets_input[j]['pT'] = events_combined[i][j*3]
              pseudojets_input[j]['eta'] = events_combined[i][j*3+1]
              pseudojets_input[j]['phi'] = events_combined[i][j*3+2]
              #print('Constituent small R jet pt: ', pseudojets_input[j]['pT'])
              pass
          pass
    
      sequence = cluster(pseudojets_input, R=1.0, p=-1)
      jets = sequence.inclusive_jets(ptmin=20)   #resulting clustered jets with pT > 20
 
      ############################
      ### Lund plane
      ############################
      for jet in jets: 
       
        #print('Jet pt: ' , jet.pt, ", jet mass: ", jet.mass)
        if jet.pt < 150 or jet.mass < 50: 
          #print('Low pT/low mass jet, not processing')
          continue
        else:
          #do lund plane
          tmp_lund = calc_lund_planes(jet)

          #push to respective groups (signal, validation, training)
          if issignal: 
            lund_planes.append(tmp_lund)
        pass
 
        if len(lund_planes) >= 0: 
          sorted_lund_planes[str(n_c)].append(lund_planes) 

  #------------------------------------------------------------------------------------------------------------------------------------
  #------------------------------------------------------------------------------------------------------------------------------------
  outfile_sig = h5py.File("events_anomalydetection_VRNN_lundPlanes.hdf5","w")
  for i in range(n_start,n_consts):
    outfile_sig.create_dataset(str(i)+"/hlvs", data=sorted_lund_planes[str(i)])
  outfile_sig.close()
