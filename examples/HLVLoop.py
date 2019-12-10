import h5py    
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from skhep.math.vectors import *
from scipy.special import softmax
import pickle
import glob
from pprint import pprint

#import ROOT
#ROOT.PyConfig.IgnoreCommandLineOptions = True
#ROOT.gROOT.SetBatch(1)
#from ROOT import THStack, TH1F, TF1, TCanvas, gStyle, TLatex, TColor, TLegend, TGraph, TGraphErrors, TVector, TVectorT, gPad, TLorentzVector, TMatrixDSym, TMatrixDSymEigen

import sys 
sys.path.append("/Users/juliagonski/Documents/Columbia/Physics/yXH/test_pyjet_extfastjet/pyjet")
from pyjet import DTYPE_PTEPM,ClusterSequence,JetDefinition,PseudoJet,cluster,EnergyCorrelator,EnergyCorrelatorC2,EnergyCorrelatorD2,Nsubjettiness,NsubjettinessRatio,KT_Axes,NormalizedMeasure

#using namespace fastjet

################################### 
## Substructure Variables 
#---- Energy Correlator
# double EnergyCorrelator::result(const PseudoJet& jet) const 
#-------------------------------------------------------------------
def calc_ecf(jet):
  #From fjcontrib EnergyCorrelator.cc: observable C2 ECF(3,beta)*ECF(1,beta)/ECF(2,beta)^2 
  #From fjcontrib EnergyCorrelator.cc: observable D2 ECF(3,beta)*ECF(1,beta)^3/ECF(2,beta)^3 
  beta = 1.0 #"most common use of N-subjettiness in the literature takes beta = 1"
  measure = 'pt_R' #enum; 
  strategy = 'storage_array' #enum; or 'slow' 
  ECF1 = EnergyCorrelator(1,beta)
  ECF2 = EnergyCorrelator(2,beta)
  ECF3 = EnergyCorrelator(3,beta)
  result_1 = ECF1.result(jet)
  result_2 = ECF2.result(jet)
  result_3 = ECF3.result(jet)
  #Confirmed that these match
  #C2:
  #c2 = result_3*result_1/np.power(result_2,2)
  #print('C2: ', c2)
  #D2: 
  #d2 = np.divide(result_3*np.power(result_1,3), np.power(result_2,3))
  #print('D2: ', d2)
  # fjcontrib actually just has a function! 
  ECF_c2 = EnergyCorrelatorC2(beta, measure, strategy) 
  ECF_d2 = EnergyCorrelatorD2(beta, measure, strategy) 
  #print('From dedicated tool: ', ECF_c2.result(jet), ECF_d2.result(jet))
  return [ECF_c2.result(jet), ECF_d2.result(jet)]


#---- Nsubjettiness
#-------------------------------------------------------------------
def calc_tau(jet):
  #axes_def = contrib::KT_Axes() #from athena 
  #measure_def= contrib::NormalizedMeasure() #these are classes...  fastjet::contrib::KT_Axes kt_axes
  axes_def = KT_Axes()
  measure_def = NormalizedMeasure(1.0,1.0) #beta, R0
  Nsub_1 = Nsubjettiness(1,axes_def,measure_def)
  Nsub_2 = Nsubjettiness(2,axes_def,measure_def)
  Nsub_3 = Nsubjettiness(3,axes_def,measure_def)
  tau_1 = Nsub_1.result(jet)
  tau_2 = Nsub_2.result(jet)
  tau_3 = Nsub_3.result(jet)
  #print('Tau1: ' , tau_1, ', tau_2: ' , tau_2, ', tau_3: ', tau_3)

  return [tau_1,tau_2,tau_3]
#-------------------------------------------------------------------
def calc_tauratio(jet):
  axes_def = KT_Axes()
  measure_def = NormalizedMeasure(1.0,1.0)
  Nsub_21 = NsubjettinessRatio(2,1,axes_def,measure_def)
  Nsub_32 = NsubjettinessRatio(3,2,axes_def,measure_def)
  Nsub_31 = NsubjettinessRatio(3,1,axes_def,measure_def)
  tau_21 = Nsub_21.result(jet)
  tau_32 = Nsub_32.result(jet)
  tau_31 = Nsub_31.result(jet)
  #print('Tau21: ' , tau_21, ', tau_23: ' , tau_23, ', tau_13: ', tau_13)

  return [tau_21,tau_32,tau_31]


#---- Kt splitting
#-------------------------------------------------------------------
def calc_ktsplit(jet):
  split12 = -1 
  split23 = -1

  #ekt_jd = JetDefinition('kt',1.5) #E_scheme,Best)
  #kt_seq_excl = ClusterSequence(jet.constituents_array(), R=1.5, p=1)
  kt_seq_excl = cluster(jet.constituents_array(), R=1.5, p=1)
  old_kt_jets = kt_seq_excl.inclusive_jets() #large R jets
  old_kt_jets.sort() #sorted backwards
  kt_jets = np.flip(old_kt_jets)
  kt_jet = kt_jets[0]
  #print('kt jet: ' , kt_jet)
  split12 = 1.5*sqrt(kt_seq_excl.exclusive_subdmerge(kt_jet, 1))
  split23 = 1.5*sqrt(kt_seq_excl.exclusive_subdmerge(kt_jet, 2))
  #print('Split12: ' , split12, ', split23: ' , split23)

  return [split12, split23]


#---- Simple vars
#-------------------------------------------------------------------
def boost(jet, bx, by, bz):
   b2 = np.power(bx,2) + np.power(by,2) + np.power(bz,2)
   gamma = np.divide(1.,np.sqrt( 1. - b2 ))
   bp = bx * jet.px + by * jet.py + bz * jet.pz
   if b2 > 0.: gamma2 = np.divide(gamma - 1.0, b2)
   else: gamma2 = 0.

   xp = jet.px + gamma2 * bp * bx + gamma * bx * jet.t
   yp = jet.py + gamma2 * bp * by + gamma * by * jet.t
   zp = jet.pz + gamma2 * bp * bz + gamma * bz * jet.t
   tp = gamma * ( jet.t + bp )

   return LorentzVector(xp, yp, zp, tp)

#-------------------------------------------------------------------
def boost_to_center_of_mass(jet):

  clusters = []
  if jet.e < 1e-20:#FPE
    return clusters

  bx = np.divide(jet.px, jet.e)
  by = np.divide(jet.py, jet.e)
  bz = np.divide(jet.pz, jet.e)
  #print('bx ' , bx, ' by,  ' , by, ' bz ', bz)

  if bx*bx + by*by + bz*bz >= 1:  # Faster than light
    return clusters

  constit_pseudojets = jet.constituents()
  for i1 in range(len(constit_pseudojets)):
    v = LorentzVector(constit_pseudojets[i1].px, constit_pseudojets[i1].py,constit_pseudojets[i1].pz,constit_pseudojets[i1].e)
    #print('LorentzVector: ' , v)
    v = boost(v, -bx,-by,-bz)
    #print('LorentzVector after boosting: ' , v)
    clusters.append(v)

  return clusters

#-------------------------------------------------------------------
def calc_aplanarity(jet):
  Aplanarity = -999.*1000

  clusters = boost_to_center_of_mass(jet) 
  if len(clusters) < 2: return Aplanarity

  MomentumTensor = np.empty((3,3))
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
    MomentumTensor[0,0] = val00/P2Sum
    MomentumTensor[1,0] = val10/P2Sum
    MomentumTensor[2,0] = val20/P2Sum
    MomentumTensor[0,1] = val01/P2Sum
    MomentumTensor[1,1] = val11/P2Sum
    MomentumTensor[2,1] = val21/P2Sum
    MomentumTensor[0,2] = val02/P2Sum
    MomentumTensor[1,2] = val12/P2Sum
    MomentumTensor[2,2] = val22/P2Sum

    u, Lambda, vh = np.linalg.svd(MomentumTensor) #unitary arrays, singular values, hermitian unitary
    #print(u, s, vh)
    Aplanarity = 1.5*Lambda[2];

  return Aplanarity

#-------------------------------------------------------------------
def calc_planarflow(jet):
  PF = -1.0
  if jet.mass == 0 or len(jet.constituents_array()) == 0: return PF
  constit_pseudojets = jet.constituents()

  #MomentumTensor = np.empty((2,2))
  MomentumTensor = np.zeros(shape=[2,2])
  #MomentumTensor = TMatrixDSym(2)
  #Planar flow
  phi0=jet.phi
  eta0=jet.eta

  nvec = np.zeros(shape=[3])
  nvec[0]=np.divide(np.cos(phi0), np.cosh(eta0))
  nvec[1]=np.divide(np.sin(phi0), np.cosh(eta0))
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

  RotationMatrix[0,0] =- ctheta0*cphi0
  RotationMatrix[0,1] =- ctheta0*sphi0
  RotationMatrix[0,2] = stheta0
  RotationMatrix[1,0] = sphi0
  RotationMatrix[1,1] =- 1.*cphi0
  RotationMatrix[1,2] = 0.
  RotationMatrix[2,0] = stheta0*cphi0
  RotationMatrix[2,1] = stheta0*sphi0
  RotationMatrix[2,2] = ctheta0 
  #print('Rotation Matrix:', RotationMatrix)

  val00 = 0.0; val10 = 0.0; val01 = 0.0; val11 = 0.0
  for cp in constit_pseudojets:
    p = LorentzVector(cp.px,cp.py,cp.pz,cp.e)
    n=1./(cp.e*jet.mass)
    px_rot = RotationMatrix[0,0] * (p.px) +RotationMatrix[0,1] * (p.py)+RotationMatrix[0,2]*(p.pz)
    py_rot = RotationMatrix[1,0] * (p.px) +RotationMatrix[1,1] * (p.py)+RotationMatrix[1,2]*(p.pz)
    pz_rot = RotationMatrix[2,0] * (p.px) +RotationMatrix[2,1] * (p.py)+RotationMatrix[2,2]*(p.pz)

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
  #print('Momentum Tensor:', MomentumTensor)

  #eigen = TMatrixDSymEigen(MomentumTensor)
  #Lambda = eigen.GetEigenValues();
  Lambda, eigen = np.linalg.eig(MomentumTensor)
  num = 4*Lambda[0]*Lambda[1]
  den = (Lambda[0]+Lambda[1]) * (Lambda[0]+Lambda[1])
  if np.abs(den) < 1.e-20: return PF
  PF = num/den
  return PF

#-------------------------------------------------------------------
def calc_qw(jet):
#### this so far is only implemented in "normal mode": 
#### https://gitlab.cern.ch/atlas/athena/blob/21.2/Reconstruction/Jet/JetSubStructureUtils/Root/Qw.cxx#L68

  qw = -1
  constituents = jet.constituents_array()
  if len(constituents)< 3: return 0

  #Build the subjets 
  cs = cluster(constituents, R=1.0, p=-1)
  scaleF = 1.
  outjets= cs.exclusive_jets(3) 
  #qw agreement is much better when all 3 outjets have nonzero mas... 
  #print('outjets: ' , outjets)
  m0_jet = LorentzVector(outjets[0].px, outjets[0].py, outjets[0].pz, outjets[0].e)
  m1_jet = LorentzVector(outjets[1].px, outjets[1].py, outjets[1].pz, outjets[1].e)
  m2_jet = LorentzVector(outjets[2].px, outjets[2].py, outjets[2].pz, outjets[2].e)

  m12 = (m0_jet + m1_jet).m
  m23 = (m2_jet + m1_jet).m
  m13 = (m2_jet + m0_jet).m 
  #print('m12: ' , m12, ', m23: ' , m23, ' , m13: ' , m13)

  qw = scaleF*np.minimum( m12, np.minimum(m23,m13) )
  #print('Qw: ' , qw)

  return qw

#-------------------------------------------------------------------
def calc_zcut(jet):
  #if Split12 and Split23 agree, zcut tends to as well... 
  #dmin12  = jet_AntiKt10LCTopo_SPLIT12->at(i);
  #jetmass = jet_AntiKt10LCTopo_constscale_m->at(i);
  #zcut12  = std::pow(dmin12,2)/(std::pow(dmin12,2)+std::pow(jetmass,2));


  constit_pseudojets = jet.constituents_array()
  if len(constit_pseudojets) == 0: return -1
  m_nSubJets = 1 #12 variables
  kt_clust_seq = cluster(constit_pseudojets, R=1.5, p=1)

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
  #print('last split subjet: ' , lastSplitSubjet)

  if max_cluster_hist_index == -1: return 0.0 #None of the subjets were split

  #split = KtSplittingScale(m_nSubJets)
  old_kt_jets = kt_clust_seq.inclusive_jets()
  old_kt_jets.sort() #sorted backwards
  kt_jets = np.flip(old_kt_jets)
  kt_jet = kt_jets[0]
  #print('kt jet: ' , kt_jet)
  split = 1.5*np.sqrt(kt_clust_seq.exclusive_subdmerge(kt_jet, m_nSubJets))
  dmin = pow(split, 2.0)

  zcut = -1
  #print('dmin: ', dmin, ', last split subjet mass squared: ' , lastSplitSubjet.m2)
  if dmin == 0: zcut = 0
  else: zcut = np.divide(dmin, dmin + lastSplitSubjet.m2)

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
    #t_theta_i = t_jet_p4.Angle(t_tclus.Vect())
    vec1 = [tclu.px,tclu.py,tclu.pz]
    vec2 = [jet.px, jet.py, jet.pz]
    theta_i=np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))

    sintheta_i = sin(theta_i)
    if sintheta_i == 0: continue #avoid FPE
    e_theta_i_a2 = tclu.e*pow(sintheta_i,m_a2)*pow(1-cos(theta_i),1-m_a2)
    sum_a2 += e_theta_i_a2

  if jet.mass < 1.e-20: return -1.0
  Angularity2 = sum_a2/jet.mass 
  return Angularity2



#-------------------------------------------------------------------
def calc_ktdr(jet):

  jets = jet.constituents_array()
  if len(jets) < 2: return 0.0
  #sequence = cluster(jets, R=1.0, p=-1)
  jd = JetDefinition('kt',1.0) #E_scheme,Best)
  sequence = cluster(jets, jd)
  jets_sub = sequence.exclusive_jets(2)   #constituent list 
  if len(jets_sub) < 2: return 0.0

  twopi = 2.0*np.arccos(-1.0)
  deta = jets_sub[1].eta - jets_sub[0].eta
  dphi = np.absolute(jets_sub[1].phi - jets_sub[0].phi)
  if dphi > twopi: dphi -= twopi
  dr = np.sqrt(deta*deta + dphi*dphi)
  return dr

###############################################################################################
###############################################################################################
#-------------------------------------------------------------------
def soft_pts(constituents):
  if len(constituents)>0:
    pts = np.asarray(constituents).transpose()[0]
    try:
        #soft_pts = np.divide(pts, jet_final.px)
        soft_pts = np.divide(pts, np.sum(pts))
    except ValueError:
        print("Divide by zero error in soft pts")
    for i in range(n_c):
      constituents[i][0] = soft_pts[i]

  return constituents

#-------------------------------------------------------------------
def boost_jet(Jet, n_consts, constituents):
  n_c_hist = []
  n_c_nocut_hist = []
  #print('For this jet, num constituents : ' , len(Jet))
  #for constit in Jet: print('Constituent: ', constit)

  #######################################################
  ##### ------------ Large R jet handling
  #######################################################

  jet = LorentzVector()
  jet.setptetaphim(Jet.pt, Jet.eta, Jet.phi, Jet.mass)
  print("Large R jet four vector: " , jet.pt, jet.eta, jet.phi, jet.m)

  #######################################################
  #### ------------ Constituent handling
  #######################################################

  #for constit in Jet: print('Constituent: ', constit)
  old_constits = Jet.constituents_array() 
  old_constits.sort() #sorted backwards
  constits = np.flip(old_constits)

  #constituents = np.zeros(shape=[n_consts, 3])
  #constituents = [] 
  n_c = 0
  n_c_nocut = 0

  #-- Loop through jet constituents
  #for j in range(1, n_consts+1):
  upper_n_consts = 80
  row_largerjet = [Jet.pt, Jet.eta, Jet.phi, Jet.mass]
  #if n_consts > len(constits): upper_n_consts = len(constits) - 1
  #print('Upper end of n_consts: ', upper_n_consts)
  constituents.append(row_largerjet)
  print('Constits large r only: ', constituents)
  for j in range(0, upper_n_consts): #need to reverse this?! 
    row = []
    n_c_nocut += 1
    cst = LorentzVector()
    #cst.setptetaphim(constits[j][0], constits[j][1], constits[j][2], constits[j][3])
    if j >= len(constits): row = [0,0,0,0]
    else: row = [constits[j][0], constits[j][1], constits[j][2], constits[j][3]]
    constituents.append(row)

  print('Constits after adding: ', constituents)
  n_c_hist.append(n_c)
  n_c_nocut_hist.append(n_c_nocut)

  #Softmax PTs
  #print(constituents)
    #Add to sorted array
    #new_constituents.append(constituents)

  return constituents  


#-------------------------------------------------------------------------
if __name__ == "__main__":

  f = pd.read_hdf("/Users/juliagonski/Documents/Columbia/Physics/yXH/lhcOlympics2020/events_anomalydetection.h5")
  events_combined = f.T
  np.shape(events_combined)

  #n_consts = 11
  #n_start = 10
  n_consts = 1
  n_start = 0


  #Now, let's cluster some jets!
  sorted_constituents_even = []
  sorted_constituents_odd = []
  sorted_constituents_signal = []
  sorted_hlvs_even = []
  sorted_hlvs_odd = []
  sorted_hlvs_signal = []
  #for i in range(n_start,n_consts):
  #  sorted_constituents_signal.update({str(i):[]})
  #  sorted_constituents_even.update({str(i):[]})
  #  sorted_constituents_odd.update({str(i):[]})
  #  sorted_hlvs_signal.update({str(i):[]})
  #  sorted_hlvs_even.update({str(i):[]})
  #  sorted_hlvs_odd.update({str(i):[]})

  for n_c in range(n_start, n_consts):
    print('Current n_Consts: ', n_c)
    alljets = {}

    #for mytype in ['background','signal']:
    #leadpT[mytype]=[]
    #alljets[mytype]=[]
    for i in range(1000): #len(events_combined)):
      constituents_signal_tmp = []
      constituents_odd_tmp = []
      constituents_even_tmp = []
      hlvs_signal_tmp = []
      hlvs_odd_tmp = []
      hlvs_even_tmp = []
      if (i%10000==0):
          print(i)
          pass
      issignal = events_combined[i][2100]
      if issignal: print('Signal event ', i)
      elif i%2 == 1: print('Even event ', i)
      else: print('Odd event ', i)
      #if (mytype=='background' and issignal):
      #    continue
      #elif (mytype=='signal' and issignal==0):
      #     continue
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
 
      #########################
      ### Boost jets 
      #########################
      print('For this event, number of clustered large-R jets: ', len(jets))
      for jj in range(10): 
        print('New jet! ', jj)
        if jj >= len(jets):    
          print('Starting to zero pad')
          tmp_hlvs = np.zeros((14))
          constituents_signal = np.zeros((81,4))
          constituents_even = np.zeros((81,4))
          constituents_odd = np.zeros((81,4))
        else:
          consts_to_add = []
          constituents_signal = []
          constituents_odd = []
          constituents_even = []
          hlvs_signal = []
          hlvs_odd = []
          hlvs_even = []

          #print('Jet pt: ' , jet.pt, ", jet mass: ", jet.mass)
          row_largerjet = [jets[jj].pt, jets[jj].eta, jets[jj].phi, jets[jj].mass]
          #if jet.pt < 150 or jet.mass < 50: 
          if jets[jj].pt < 0 or jets[jj].mass < 0: 
            print('NEGATIVE pT/low mass jet, not processing')
            print('this weird jet: ', jets[jj])
            tmp_hlvs = np.zeros((14))
            constituents_signal = np.zeros((81,4))
            constituents_even = np.zeros((81,4))
            constituents_odd = np.zeros((81,4))
            #continue
          else:
            #calc substructure variables 
            #ECF
            c2, d2 = calc_ecf(jets[jj])
            #Nsubjettiness
            tau1,tau2,tau3 = calc_tau(jets[jj])
            tau21,tau32,tau31 = calc_tauratio(jets[jj])
            aplanarity = calc_aplanarity(jets[jj])
            #Kt splitting
            split12,split23 = calc_ktsplit(jets[jj])
            #Simple vars
            KtDR = calc_ktdr(jets[jj])
            planarFlow = calc_planarflow(jets[jj]) #NOT WORKING
            angularity = calc_angularity(jets[jj])
            zcut = calc_zcut(jets[jj])
            qw = calc_qw(jets[jj]) #NOT WORKING!
            tmp_hlvs = [c2,d2,tau1,tau2,tau3,tau21,tau32,tau31,aplanarity,split12,split23,angularity,KtDR,zcut]

          #print('       My script; ')
          #print('c2: ', c2)
          #print('d2: ', d2)   
          #print('tau1: ', tau1)
          #print('tau2: ', tau2)  
          #print('tau3: ', tau3)  
          #print('tau21: ', tau21) 
          #print('tau32: ', tau32) 
          #print('tau31: ', tau31) 
          #print('Aplanarity: ', aplanarity)
          #print('Split12: ', split12)
          #print('Split23: ', split23)
          #print('KtDR: ', KtDR)
          #print('Planar flow: ', planarFlow)
          #print('Angularity: ', angularity)
          #print('Zcut12: ', zcut)
          #print('Qw: ', qw)

          #push to respective groups (signal, validation, training)
          if issignal: 
            #print('Type of event = signal')
            if jj >= len(jets) or jets[jj].pt < 0 or jets[jj].mass < 0: print('zero pad')
            else: 
              #if jj==0: constituents_signal.append(row_largerjet)
              constituents_signal = boost_jet(jets[jj], n_c, constituents_signal)
            hlvs_signal = tmp_hlvs
          else: #split background into training + validation
            if i%2 == 1: 
              #print('Type of event = even')
              if jj >= len(jets) or jets[jj].pt < 0 or jets[jj].mass < 0: print('zero pad')
              else: 
                #if jj==0: constituents_even.append(row_largerjet)
                constituents_even = boost_jet(jets[jj], n_c, constituents_even)
              hlvs_even = tmp_hlvs
            else: 
              #print('Type of event = odd')
              if jj >= len(jets) or jets[jj].pt < 0 or jets[jj].mass < 0: print('zero pad')
              else: 
                #if jj==0: constituents_odd.append(row_largerjet)
                constituents_odd = boost_jet(jets[jj], n_c, constituents_odd)
              hlvs_odd = tmp_hlvs

        #print('hlvs signal: ' , hlvs_signal)
        #print('hlvs even: ' , hlvs_even)
        #print('hlvs odd: ' , hlvs_odd)
        #print('constituents signal: ' , constituents_signal)
        #print('constituents even: ' , constituents_even)
        #print('constituents odd: ' , constituents_odd)

        pass
 

        if len(constituents_signal) >= n_c:  #zero pad these arrays!
          constituents_signal_tmp.append(constituents_signal)
          hlvs_signal_tmp.append(hlvs_signal) 
        if len(constituents_even) >= n_c: 
          constituents_even_tmp.append(constituents_even)
          hlvs_even_tmp.append(hlvs_even) 
        if len(constituents_odd) >= n_c: 
          constituents_odd_tmp.append(constituents_odd)
          hlvs_odd_tmp.append(hlvs_odd)
      
      #print('BEFORE Sorted constituents signal event : ', np.array(sorted_constituents_signal).shape)
      #print('BEFORE Sorted constituents even event: ',   np.array(sorted_constituents_even).shape)
      #print('BEFORE Sorted constituents odd event: ',    np.array(sorted_constituents_odd).shape)
      if issignal: 
        sorted_constituents_signal.append(constituents_signal_tmp)
        sorted_hlvs_signal.append(hlvs_signal_tmp) 
      elif i%2 == 1:
        sorted_constituents_even.append(constituents_even_tmp)
        sorted_hlvs_even.append(hlvs_even_tmp) 
      else:
        sorted_constituents_odd.append(constituents_odd_tmp)
        sorted_hlvs_odd.append(hlvs_odd_tmp)

      print('AFTER Sorted constituents signal event : ',np.array(sorted_constituents_signal).shape)
      print('AFTER Sorted constituents even event: ',   np.array(sorted_constituents_even).shape)
      print('AFTER Sorted constituents odd event: ',    np.array(sorted_constituents_odd).shape)
      if i > 30 and (np.ndim(np.array(sorted_constituents_signal)) < 4 or np.ndim(np.array(sorted_constituents_even)) < 4 or np.ndim(np.array(sorted_constituents_odd)) < 4): 
        print('Broken, exit "gracefully"')
        break
      print('HLVs signal event : ', np.array(sorted_hlvs_signal).shape)
      print('HLVs even event: ',   np.array(sorted_hlvs_even).shape)
      print('HLVs odd event: ',    np.array(sorted_hlvs_odd).shape)
      if i > 30 and (np.ndim(np.array(sorted_hlvs_signal)) < 3 or np.ndim(np.array(sorted_hlvs_even)) < 3 or np.ndim(np.array(sorted_hlvs_odd)) < 3): 
        print('Broken, exit "gracefully"')
        break

    ### end loop over events



  #------------------------------------------------------------------------------------------------------------------------------------
  #------------------------------------------------------------------------------------------------------------------------------------
  outfile_valid = h5py.File("events_anomalydetection_validation_boosted_VRNN.hdf5","w")
  outfile_valid.create_dataset(str(i)+"/constituents", data=sorted_constituents_odd)
  outfile_valid.create_dataset(str(i)+"/hlvs", data=sorted_hlvs_odd)
  outfile_valid.close()

  outfile_train = h5py.File("events_anomalydetection_training_boosted_VRNN.hdf5","w")
  outfile_train.create_dataset(str(i)+"/constituents", data=sorted_constituents_even)
  outfile_train.create_dataset(str(i)+"/hlvs", data=sorted_hlvs_even)
  outfile_train.close()

  outfile_sig = h5py.File("events_anomalydetection_signal_boosted_VRNN.hdf5","w")
  outfile_sig.create_dataset(str(i)+"/constituents", data=sorted_constituents_signal)
  outfile_sig.create_dataset(str(i)+"/hlvs", data=sorted_hlvs_signal)
  outfile_sig.close()


  #f = open("FTAG_LSTM.pkl", "wb")
  #pickle.dump(sorted_constituents, f)
  #f.close()

  #for i in range(n_consts+1):
  #  if(len(sorted_constituents[str(i)]) > 0):
  ##    np.save("Sorted_Constituents/test_sorted_"+str(i)+".npy", sorted_constituents[str(i)])
  #    print(str(i) + " Consts: " + str(len(sorted_constituents[str(i)])))

  #print(sorted_constituents["2"])
  #print(np.shape(sorted_constituents["2"]))
  #np.save("test_sorted.npy", sorted_constituents[2])
  #outfile = h5py.File("events_boostedconstit.hdf5")

  #Let's make some very simple plots.
  #fig = plt.figure()
  #ax = fig.add_subplot(1, 1, 1)
  #n,b,p = plt.hist(leadpT['background'], bins=50, facecolor='r', alpha=0.2,label='background')
  #plt.hist(leadpT['signal'], bins=b, facecolor='b', alpha=0.2,label='signal')
  #plt.xlabel(r'Leading jet $p_{T}$ [GeV]')
  #plt.ylabel('Number of events')
  #plt.legend(loc='upper right')
  ##plt.show()
  #plt.savefig("plots/leadjetpt.pdf")
  #
  #
  #mjj={}
  #for mytype in ['background','signal']:
  #    mjj[mytype]=[]
  #    for k in range(len(alljets[mytype])):
  #        E = alljets[mytype][k][0].e+alljets[mytype][k][1].e
  #        px = alljets[mytype][k][0].px+alljets[mytype][k][1].px
  #        py = alljets[mytype][k][0].py+alljets[mytype][k][1].py
  #        pz = alljets[mytype][k][0].pz+alljets[mytype][k][1].pz
  #        mjj[mytype]+=[(E**2-px**2-py**2-pz**2)**0.5]
  #        pass
  #    pass
  #
  #fig = plt.figure()
  #ax = fig.add_subplot(1, 1, 1)
  #n,b,p = plt.hist(mjj['background'], bins=50, facecolor='r', alpha=0.2,label='background')
  #plt.hist(mjj['signal'], bins=b, facecolor='b', alpha=0.2,label='signal')
  #plt.xlabel(r'$m_{JJ}$ [GeV]')
  #plt.ylabel('Number of events')
  #plt.legend(loc='upper right')
  ##plt.show()
  #plt.savefig("plots/mjj.pdf")

