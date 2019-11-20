###### To implement changes in fastjet: 
#-- (pip install cython on xenia)
#-- Add functionality from fastjet to fastjet.pxd file
#-- in pyjet/src: cython —cplus __libpyjet.pyx (this generates .cpp file)
#-- in pyjet: make 

import h5py    
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from skhep.math.vectors import *
from scipy.special import softmax
import pickle
import glob
from pprint import pprint

import sys 
sys.path.append("/nevis/kolya/home/jgonski/yXH/pyjet")
from pyjet import DTYPE_PTEPM,ClusterSequence,JetDefinition,PseudoJet,cluster
#from pyjet import *


################################### 
## Substructure Variables 
#-------------------------------------------------------------------
def calc_aplanarity(jet):
  Aplanarity = -999.*1000

  #clusters = boostToCenterOfMass(jet, jet.constituents()); TODO
  if clusters.size() < 2: return Aplanarity

  MomentumTensor = TMatrixD(3,3)
  P2Sum = 0

  val00 = 0.0; val01 = 0.0; val02 = 0.0; val10 = 0.0; val11 = 0.0; val12 = 0.0; val20 = 0.0; val21 = 0.0; val22 = 0.0
  for itr in clusters:
    val00 += itr.px * itr.px
    val01 += itr.px * itr.py
    val02 += itr.px * itr.pz
    val10 += itr.py * itr.px
    val11 += itr.py * itr.py
    val12 += itr.py * itr.pz
    val20 += itr.pz * itr.px
    val21 += itr.pz * itr.py
    val22 += itr.pz * itr.pz
    P2Sum += itr.px*itr.px+ itr.py*itr.py+ itr.pz*itr.pz
  

  Aplanarity = -1;

  if P2Sum > 0:
    MomentumTensor[0,0] = val00
    MomentumTensor[1,0] = val10
    MomentumTensor[2,0] = val20
    MomentumTensor[0,1] = val01
    MomentumTensor[1,1] = val11
    MomentumTensor[2,1] = val21
    MomentumTensor[0,2] = val02
    MomentumTensor[1,2] = val12
    MomentumTensor[2,2] = val22

    aSVD = TDecompSVD(MomentumTensor)
    Lambda = aSVD.GetSig()

    Aplanarity = 1.5*Lambda[2];
    #delete aSVDA

  return Aplanarity

#-------------------------------------------------------------------
def calc_qw(jet):

  constituents = jet.constituents_array()
  if len(constituents)< 3: return 0

  #Build the subjets 
  cs = cluster(constituents, R=1.0, p=-1)
  scaleF = 1.
  outjets= cs.exclusive_jets(3) #TODO 
  #print(outjets)

  #m12 = (outjets[0]+outjets[1]).mass
  #m23 = (outjets[2]+outjets[1]).mass
  #m13 = (outjets[2]+outjets[0]).mass

  #TODO qw = scaleF*np.min(m12, np.min(m23,m13) )
  qw = -1

  return qw

#-------------------------------------------------------------------
def calc_zcut(jet):

  constit_pseudojets = jet.constituents_array()
  if len(constit_pseudojets) == 0: return -1
  m_nSubJets = 2
 
  #jet_def = JetDefinition('antikt', 1.0) 
  #kt_clust_seq = ClusterSequence(constit_pseudojets, jet_def)
  kt_clust_seq = cluster(constit_pseudojets, R=1.0, p=-1)

  if len(constit_pseudojets) < m_nSubJets: return 0.0 #We were asked to calculate zCut, but there are not enough constituents
  subjets = kt_clust_seq.exclusive_jets(m_nSubJets)

  #Find last split jet (cluster_hist_index should be highest for the last created jet)
  lastSplitSubjet = None
  max_cluster_hist_index = -1
   
  for subj in subjets:
    if subj.parents != None: 
      parent1 = subj.parents[0]; parent2 = subj.parents[1]
    if subj.parents != None and subj.cluster_hist_index > max_cluster_hist_index:
      max_cluster_hist_index = subj.cluster_hist_index
      lastSplitSubjet = subj

  if max_cluster_hist_index == -1: return 0.0 #None of the subjets were split

  #split = KtSplittingScale(m_nSubJets)
  #dmin = pow(split.result(jet), 2.0)

  zcut = -1
  #if dmin == 0: zcut = 0
  #TODO else: zcut = dmin / (dmin + lastSplitSubjet.m2())

  return zcut


#-------------------------------------------------------------------
def calc_angularity(jet):

  if len(jet.constituents_array()) == 0: return -1
  if jet.mass ==0.0: return -1

  constit_pseudojets = jet.constituents()
  jet_p4 = LorentzVector(jet.px, jet.py, jet.pz, jet.e)

  Angularity2=-1.
  m_a2=-2.
  sum_a2=0.

  for tclu in constit_pseudojets:
    tclus = LorentzVector(tclu.px,tclu.py,tclu.pz,tclu.e)
    #theta_i = jet_p4.Angle(tclus.Vect())
    v0 = [tclu.px,tclu.py,tclu.pz]
    v1 = [jet.px, jet.py, jet.pz]
    v0_u = v0 / np.linalg.norm(v0)
    v1_u = v0 / np.linalg.norm(v1)
    theta_i = np.arccos(np.clip(np.dot(v0_u, v1_u), -1.0, 1.0))
    sintheta_i = sin(theta_i)
    if sintheta_i == 0: continue #avoid FPE
    e_theta_i_a2 = tclu.e*pow(sintheta_i,m_a2)*pow(1-cos(theta_i),1-m_a2)
    sum_a2 += e_theta_i_a2

  if jet.mass < 1.e-20: return -1.0
  Angularity2 = sum_a2/jet.mass 
  return Angularity2


#-------------------------------------------------------------------
def calc_planarflow(jet):
  PF = -1.0
  if jet.mass == 0 or len(jet.constituents_array()) == 0: return PF
  constit_pseudojets = jet.constituents()

  MomentumTensor = np.empty((2,2))
  #Planar flow
  phi0=jet.phi
  eta0=jet.eta

  nvec = np.zeros(shape=[3])
  nvec[0]=(np.cos(phi0)/np.cosh(eta0))
  nvec[1]=(np.sin(phi0)/np.cosh(eta0))
  nvec[2]=np.tanh(eta0)

  #this is the rotation matrix
  RotationMatrix = np.zeros(shape=[3,3])

  mag3 = np.sqrt(nvec[0]*nvec[0] + nvec[1]*nvec[1]+ nvec[2]*nvec[2])
  mag2 = np.sqrt(nvec[0]*nvec[0] + nvec[1]*nvec[1])

  #if rotation axis is null
  if mag3 <= 0: return PF

  ctheta0 = nvec[2]/mag3
  stheta0 = mag2/mag3
  cphi0 = nvec[0]/mag2 if mag2>0. else 0.
  sphi0 = nvec[1]/mag2 if mag2>0. else 0.

  RotationMatrix[0,0] = ctheta0*cphi0
  RotationMatrix[0,0] =- ctheta0*cphi0
  RotationMatrix[0,1] =- ctheta0*sphi0
  RotationMatrix[0,2] = stheta0
  RotationMatrix[1,0] = sphi0
  RotationMatrix[1,1] =- 1.*cphi0
  RotationMatrix[1,2] = 0.
  RotationMatrix[2,0] = stheta0*cphi0
  RotationMatrix[2,1] = stheta0*sphi0
  RotationMatrix[2,2] = ctheta0

  val00 = 0.0; val10 = 0.0; val01 = 0.0; val11 = 0.0
  for cp in constit_pseudojets:
    p = LorentzVector(cp.px,cp.py,cp.pz,cp.e)
    n=1./(cp.e*jet.mass)
    px_rot = RotationMatrix[0,0] * (p.px)+RotationMatrix[0,1] * (p.py)+RotationMatrix[0,2]*(p.pz)
    py_rot = RotationMatrix[1,0] * (p.px)+RotationMatrix[1,1] * (p.py)+RotationMatrix[1,2]*(p.pz)
    pz_rot = RotationMatrix[2,0] * (p.px)+RotationMatrix[2,1] * (p.py)+RotationMatrix[2,2]*(p.pz)

    prot = LorentzVector(0.0,0.0,0.0,0.0)  
    prot.setpxpypze(px_rot, py_rot, pz_rot, p.e )

    val00 += n * prot.px * prot.px
    val01 += n * prot.py * prot.px
    val10 += n * prot.px * prot.py
    val11 += n * prot.py * prot.py
  
  MomentumTensor[0,0] = val00
  MomentumTensor[0,1] = val01
  MomentumTensor[0,0] = val10
  MomentumTensor[1,1] = val11

  #eigen = TMatrixDSymEigen(MomentumTensor)
  Lambda, eigen = np.linalg.eig(MomentumTensor)
  num = 4*Lambda[0]*Lambda[1]
  den = (Lambda[0]+Lambda[1]) * (Lambda[0]+Lambda[1])
  if np.abs(den) < 1.e-20: return PF
  PF = num/den
  return PF

#-------------------------------------------------------------------
def calc_ktdr(jet):

  jets = jet.constituents_array()
  if len(jets) < 2: return 0.0
  sequence = cluster(jets, R=1.0, p=-1)
  jets_sub = sequence.exclusive_jets(2)   #constituent list 
  if len(jets_sub) < 2: return 0.0

  twopi = 2.0*np.arccos(-1.0)
  deta = jets_sub[1].eta - jets_sub[0].eta
  dphi = np.absolute(jets_sub[1].phi - jets_sub[0].phi)
  if dphi > twopi: dphi -= twopi
  dr = np.sqrt(deta*deta + dphi*dphi)
  return dr


##########################################################################
#-------------------------------------------------------------------------
if __name__ == "__main__":

  f = pd.read_hdf("/data/users/jgonski/LHCOlympics2020/events_anomalydetection.h5")
  events_combined = f.T
  np.shape(events_combined)

  n_consts = 11
  n_start = 10


  #Now, let's cluster some jets!
  sorted_hlvs_signal = {}
  for i in range(n_start,n_consts):
    sorted_hlvs_signal.update({str(i):[]})

  for n_c in range(n_start, n_consts):
    print('Current n_Consts: ', n_c)

    #for mytype in ['background','signal']:
    for i in range(100000): #len(events_combined)):
      if (i%10000==0):
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
      jets = sequence.inclusive_jets(ptmin=20)   #constituent list 
 
      ############################
      ### Substructure variables
      ############################
      #print('Number of clustered large-R jets: ', len(jets))
      for jet in jets: 
        hlvs_signal = []
        
        #print('Jet pt: ' , jet.pt, ", jet mass: ", jet.mass)
        if jet.pt < 150 or jet.mass < 50: 
          #print('Low pT/low mass jet, not processing')
          continue
        else:

          #calc substructure variables 
          KtDR = calc_ktdr(jet)
          planarFlow = calc_planarflow(jet)
          angularity = calc_angularity(jet)
          zcut = calc_zcut(jet)
          qw = calc_qw(jet) ##do we have a massCut or SmallSubjets scenario??
          tmp_hlvs = [planarFlow, angularity, KtDR, zcut, qw]

          #push to respective groups (signal, validation, training)
          if issignal: 
            hlvs_signal.append(tmp_hlvs)
        pass
 
        if len(hlvs_signal) >= 0: 
          sorted_hlvs_signal[str(n_c)].append(hlvs_signal) 

  #------------------------------------------------------------------------------------------------------------------------------------
  #------------------------------------------------------------------------------------------------------------------------------------
  outfile_sig = h5py.File("events_anomalydetection_signal_boosted_VRNN.hdf5","w")
  for i in range(n_start,n_consts):
    outfile_sig.create_dataset(str(i)+"/hlvs", data=sorted_hlvs_signal[str(i)])
  outfile_sig.close()
