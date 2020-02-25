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
sys.path.append("/Users/juliagonski/Documents/Columbia/Physics/yXH/vrnn_anomalyTagging/test_pyjet_extfastjet/pyjet")
from pyjet import DTYPE_PTEPM,ClusterSequence,JetDefinition,PseudoJet,cluster,EnergyCorrelator,EnergyCorrelatorC2,EnergyCorrelatorD2,Nsubjettiness,NsubjettinessRatio,KT_Axes,NormalizedMeasure,LundGenerator

from JetTree import JetTree, LundImage
np.set_printoptions(threshold=sys.maxsize)


#-------------------------------------------------------------------
def calc_lund_planes(jet):
  print('Jet: ', jet)
  #LundGen = LundGenerator('cambridge')
  #declusts = LundGen(jet)
  #print('From dedicated tool: ', LundGen.result(jet))
  #return LundGen.result(jet) 
   
  tree = JetTree(jet)
  imgs_ref=np.zeros((len(events), args.npx, args.npx))
  li_gen=LundImage()
  #imgs_ref[i]=li_gen(tree)


  ############### my versions
  #jd = JetDefinition('cambridge',1.0)
  #cs2 = cluster(jet.constituents_array(), jd) 
  
  #pair = cs2.inclusive_jets()
  #print('pair: ' , pair[0])
  #j1 = pair 
  #j2 = pair 
 
  #while j1 and j2 : 
  #  #pair[0].parents(j1, j2)     #result.push_back(LundDeclustering(pair, j1, j2));

  #  ca_jets = cs2.exclusive_jets(2) #return a vector of all jets when the event is clustered (in the exclusive sense) to exactly njets.
  #  print('last kt step: ' , ca_jets)
  #  j1 = ca_jets[0]
  #  j2 = ca_jets[1]

  #  #now work out the various Lund declustering variables
  #  Delta  = np.power(j1.eta - j2.eta, 2) + np.power(j1.phi - j2.phi,2)
  #  softer_pt = j2.pt
  #  z   = softer_pt / (softer_pt + j1.pt)
  #  kt  = softer_pt * Delta
  #  #ps_ = atan2(softer_.rap()-harder_.rap(), harder_.delta_phi_to(softer_))
  #  kappa = z * Delta
  #  pair = j1

  #j1 = fj.PseudoJet()
  #j2 = fj.PseudoJet()
  #if pseudojet and pseudojet.has_parents(j1,j2):
  #    # order the parents in pt
  #    if (j2.pt() > j1.pt()):
  #        j1,j2=j2,j1
  #    # then create two new tree nodes with j1 and j2
  #    self.harder = JetTree(j1, self)
  #    self.softer = JetTree(j2, self)
  #    self.lundCoord = LundCoordinates(j1, j2)


  #result.push_back([kt, Delta])
  result = 0
  print('Result : ', result)
  return result


##########################################################################
#-------------------------------------------------------------------------
if __name__ == "__main__":

  #print(dir(pyjet))

  f = pd.read_hdf("/Users/juliagonski/Documents/Columbia/Physics/yXH/vrnn_anomalyTagging/lhcOlympics2020/events_anomalydetection.h5", start=0, stop=100)
  events_combined = f.T
  np.shape(events_combined)

  n_consts = 11
  n_start = 10


  #Now, let's cluster some jets!
  lund_planes = []
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

      imgs_ref=np.zeros((len(jets)))
      for jet in range(len(jets)): 
       
        #print('Jet pt: ' , jet.pt, ", jet mass: ", jet.mass)
        if jets[jet].pt < 150 or jets[jet].mass < 50: 
          #print('Low pT/low mass jet, not processing')
          continue
        else:
          #do lund plane
          #tmp_lund = calc_lund_planes(jet)
          tree = JetTree(jets[jet])
          #li_gen=LundImage(24, 'kt')
          #li_gen = LundImage(npxlx = 24, y_axis='kt')
          li_gen=LundImage()
          #imgs_ref[jet]=li_gen(tree)
          print('is this it? ', li_gen(tree))
          #lund_planes.append(tmp_lund)

        pass
 
        if len(lund_planes) >= 0: 
          sorted_lund_planes[str(n_c)].append(lund_planes) 

  #------------------------------------------------------------------------------------------------------------------------------------
  #------------------------------------------------------------------------------------------------------------------------------------
  outfile_sig = h5py.File("events_anomalydetection_VRNN_lundPlanes.hdf5","w")
  for i in range(n_start,n_consts):
    outfile_sig.create_dataset(str(i)+"/hlvs", data=sorted_lund_planes[str(i)])
  outfile_sig.close()
