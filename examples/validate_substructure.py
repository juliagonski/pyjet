###### To iamplement changes in fastjet: 
#a-- (pip inastall cython on xenia)
#a-- Add functionality from fastjet to fastjet.pxd file and __libpyjet.pyx
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

import argparse
import h5py    
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from skhep.math.vectors import *
#from scipy.special import softmax
import pickle
import glob
from pprint import pprint
from ROOT import TLorentzVector,TMatrixDSym,TMatrixDSymEigen,TVectorD, TH1D,TTree,TCanvas, TFile


sys.path.append("/Users/juliagonski/Documents/Columbia/Physics/yXH/test_pyjet_extfastjet/pyjet")
from pyjet import DTYPE_PTEPM,ClusterSequence,JetDefinition,PseudoJet,cluster,EnergyCorrelator,EnergyCorrelatorC2,EnergyCorrelatorD2,Nsubjettiness,NsubjettinessRatio,WTA_KT_Axes,NormalizedMeasure
#JetFFMoments


################################### 
## Substructure Variables 
################################### 

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
  axes_def = WTA_KT_Axes()
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
  axes_def = WTA_KT_Axes()
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

  jet_def = JetDefinition('kt',1.5) #E_scheme,Best)
  kt_seq_excl = ClusterSequence(jet.constituents_array(), jet_def)
  #kt_seq_excl = cluster(jet.constituents_array(), R=1.5, p=1)
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


##########################################################################
#-------------------------------------------------------------------------
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--start", default = 0, type=int, nargs='+',
                     help="start # events")
  parser.add_argument("-e", "--end", default = 100000, type=int, nargs='+',
                     help="end # events")
  args = parser.parse_args()
  startNum = args.start[0]
  endNum = args.end[0]
  print('Start: ' , startNum, ', end: ', endNum)

  #print(dir(pyjet))

  #f = pd.read_hdf("/Users/juliagonski/Documents/Columbia/Physics/yXH/lhcOlympics2020/events_anomalydetection.h5")
  #f_ref = pd.read_hdf("/Users/juliagonski/Documents/Columbia/Physics/yXH/lhcOlympics2020/user.miochoa.19650387._000001.output.h5")
  f_ref = h5py.File("examples/user.miochoa.19650387._000001.output.h5", "r")
  #f_ref = pd.read_hdf("examples/user.miochoa.19650387._000001.output.h5",start=startNum,stop=endNum)
  #f_ref = h5py.File("examples/output_parentJet_nConst200.h5", "r")
  #events_combined = f.T
  #np.shape(events_combined)

  jets_orig = f_ref["fat_jet_constituents"]
  fat_jets = f_ref["fat_jet"]

  n_consts = 11
  n_start = 10

  hists_file = TFile( 'hists_trimmedtcc.root', 'UPDATE' )


  #Now, let's cluster some jets!
  sorted_hlvs_signal = {}
  for i in range(n_start,n_consts):
    sorted_hlvs_signal.update({str(i):[]})

  for n_c in range(n_start, n_consts):
    print('Current n_Consts: ', n_c)

    #plt.hist(data, bins = np.arange(min(data), max(data)+1, 1), normed=True)
    hist_c2 = TH1D('c2','c2',100,-100,100)
    hist_d2 = TH1D('d2','d2',100,-100,100)
    hist_tau1 = TH1D('tau1','tau1',100,-100,100)
    hist_tau2 = TH1D('tau2','tau2',100,-100,100)
    hist_tau3 = TH1D('tau3','tau3',100,-100,100)
    hist_tau21 = TH1D('tau21','tau21',100,-100,100)
    hist_tau32 = TH1D('tau32','tau32',100,-100,100)
    hist_tau31 = TH1D('tau31','tau31',100,-100,100)
    hist_aplanarity = TH1D('aplanarity','aplanarity',100,-100,100)
    hist_split12 = TH1D('split12','split12',100,-100,100)
    hist_split23 = TH1D('split23','split23',100,-100,100)
    hist_ktdr = TH1D('ktdr','ktdr',100,-100,100)
    hist_planarflow = TH1D('planarflow','planarflow',100,-500,500)
    hist_angularity = TH1D('angularity','angularity',100,-100,100)
    hist_zcut = TH1D('zcut','zcut',100,-100,100)
    hist_qw = TH1D('qw','qw',100,-500,500)


    #for mytype in ['background','signal']:
    for i in range(endNum): #len(fat_jet):
      if (i%10000==0):
          print(i)
          pass

      #this is all clustering stuff
      #pseudojets_input = np.zeros(len([x for x in jets_orig[i][0][::3] if x > 0]), dtype=DTYPE_PTEPM)
      pseudojets_input = np.zeros(shape=[50], dtype=DTYPE_PTEPM)
      for j in range(50): #number of hadrons per event
          if (jets_orig[i][j]["pt"] > 0.0):   #min pT cut to enter clustering, could make this tighter
             pseudojets_input[j][0] = jets_orig[i][j]["pt"]
             pseudojets_input[j][1] = jets_orig[i][j]["eta"]
             pseudojets_input[j][2] = jets_orig[i][j]["phi"]
             pseudojets_input[j][3] = 0 #all 0 mass?
             #print('Constituent small R jet pt: ', pseudojets_input[j][0])
          pass
      pass
    
      sequence = cluster(pseudojets_input, R=1.0, p=-1)
      jets = sequence.inclusive_jets(ptmin=0.0)   #resulting clustered jets with pT > 20
 
      ############################
      ### Substructure variables
      ############################
      #print('Number of clustered large-R jets: ', len(jets), " for event :", i)
      for jet in jets: 
        #print('pt: ', Jet["pt"], ', eta: ' , Jet["eta"], ', phi:' , Jet["phi"], ', mass: ' , Jet["mass"])

        hlvs_signal = []

        if jet.pt < 150 or jet.mass < 50: 
          #print('Low pT/low mass jet, not processing')
          continue
        else:
          #calc substructure variables 
          #ECF
          C2, D2 = calc_ecf(jet)
          #Nsubjettiness
          Tau1_wta,Tau2_wta,Tau3_wta = calc_tau(jet)
          Tau21_wta,Tau32_wta,Tau31_wta = calc_tauratio(jet)
          Aplanarity = calc_aplanarity(jet)
          #Kt splitting
          Split12,Split23 = calc_ktsplit(jet)
          #Simple vars
          KtDR = calc_ktdr(jet)
          PlanarFlow = calc_planarflow(jet)
          Angularity = calc_angularity(jet)
          ZCut12 = calc_zcut(jet)
          Qw = calc_qw(jet) ##do we have a massCut or SmallSubjets scenario??
          #tmp_hlvs = [c2,d2,tau1,tau2,tau3,tau21,tau32,tau31,aplanarity,split12,split23, planarFlow, angularity, KtDR, zcut, qw]


          #d_c2 = c2 - fat_jets[i]["C2"]
          #d_d2 = d2 - fat_jets[i]["D2"]
          #d_d2 = d2 - fat_jets[i]["D2"]
          print('       My script;            original:')
          print('c2: ', C2, ',    ', fat_jets[i]["C2"])
          print('d2: ', D2, ',    ', fat_jets[i]["D2"])
          print('tau1: ', Tau1_wta, ',    ', fat_jets[i]["Tau1_wta"])
          print('tau2: ', Tau2_wta, ',    ', fat_jets[i]["Tau2_wta"])
          print('tau3: ', Tau3_wta, ',    ', fat_jets[i]["Tau3_wta"])
          print('tau21: ', Tau21_wta, ',    ', fat_jets[i]["Tau21_wta"])
          print('tau32: ', Tau32_wta, ',    ', fat_jets[i]["Tau32_wta"])
          print('tau31: ', Tau31_wta, ',    ', fat_jets[i]["Tau31_wta"])
          print('Aplanarity: ', Aplanarity, ',        ', fat_jets[i]["Aplanarity"])
          print('Split12: ', Split12, ',     ', fat_jets[i]["Split12"])
          print('Split23: ', Split23, ',     ', fat_jets[i]["Split23"])
          print('KtDR: ', KtDR, ',         ', fat_jets[i]["KtDR"])
          print('Planar flow: ', PlanarFlow, ',    ', fat_jets[i]["PlanarFlow"])
          print('Angularity: ', Angularity, ',    ', fat_jets[i]["Angularity"])
          print('Zcut12: ', ZCut12, ',       ', fat_jets[i]["ZCut12"])
          print('Qw: ', Qw, ',    ', fat_jets[i]["Qw"])

           
          hist_c2.Fill(100*np.divide(C2-fat_jets[i]["C2"],fat_jets[i]["C2"]))
          hist_d2.Fill(100*np.divide(D2-fat_jets[i]["D2"],fat_jets[i]["D2"]))
          hist_tau1.Fill(100*np.divide(Tau1_wta-fat_jets[i]["Tau1_wta"],fat_jets[i]["Tau1_wta"]))
          hist_tau2.Fill(100*np.divide(Tau2_wta-fat_jets[i]["Tau2_wta"],fat_jets[i]["Tau2_wta"]))
          hist_tau3.Fill(100*np.divide(Tau3_wta-fat_jets[i]["Tau3_wta"],fat_jets[i]["Tau3_wta"]))
          hist_tau21.Fill(100*np.divide(Tau21_wta-fat_jets[i]["Tau21_wta"],fat_jets[i]["Tau21_wta"]))
          hist_tau32.Fill(100*np.divide(Tau32_wta-fat_jets[i]["Tau32_wta"],fat_jets[i]["Tau32_wta"]))
          hist_tau31.Fill(100*np.divide(Tau31_wta-fat_jets[i]["Tau31_wta"],fat_jets[i]["Tau31_wta"]))
          hist_split12.Fill(100*np.divide(Split12-fat_jets[i]["Split12"],fat_jets[i]["Split12"]))
          hist_split23.Fill(100*np.divide(Split23-fat_jets[i]["Split23"],fat_jets[i]["Split23"]))
          hist_angularity.Fill(100*np.divide(Angularity-fat_jets[i]["Angularity"],fat_jets[i]["Angularity"]))
          hist_aplanarity.Fill(100*np.divide(Aplanarity-fat_jets[i]["Aplanarity"],fat_jets[i]["Aplanarity"]))
          hist_ktdr.Fill(100*np.divide(KtDR-fat_jets[i]["KtDR"],fat_jets[i]["KtDR"]))
          hist_planarflow.Fill(100*np.divide(PlanarFlow-fat_jets[i]["PlanarFlow"],fat_jets[i]["PlanarFlow"]))
          hist_zcut.Fill(100*np.divide(ZCut12-fat_jets[i]["ZCut12"],fat_jets[i]["ZCut12"]))
          hist_qw.Fill(100*np.divide(Qw-fat_jets[i]["Qw"],fat_jets[i]["Qw"]))
          #print('fill c2: ',                   np.divide(c2-fat_jets[i]["C2"],fat_jets[i]["C2"]))                                  
          #print('fill d2: ',                   np.divide(d2-fat_jets[i]["D2"],fat_jets[i]["D2"]))
          #print('fill tau1: ',                 np.divide(tau1-fat_jets[i]["Tau1_wta"],fat_jets[i]["Tau1_wta"]))
          #print('fill tau2: ',                 np.divide(tau2-fat_jets[i]["Tau2_wta"],fat_jets[i]["Tau2_wta"]))
          #print('fill tau3: ',                 np.divide(tau3-fat_jets[i]["Tau3_wta"],fat_jets[i]["Tau3_wta"]))
          #print('fill tau21: ',                np.divide(tau21-fat_jets[i]["Tau21_wta"],fat_jets[i]["Tau21_wta"]))
          #print('fill tau32: ',                np.divide(tau32-fat_jets[i]["Tau32_wta"],fat_jets[i]["Tau32_wta"]))
          #print('fill tau31: ',                np.divide(tau31-fat_jets[i]["Tau31_wta"],fat_jets[i]["Tau31_wta"]))
          #print('fill Split12:',            np.divide(split12-fat_jets[i]["Split12"],fat_jets[i]["Split12"]))
          #print('fill Split23: ',              np.divide(split23-fat_jets[i]["Split23"],fat_jets[i]["Split23"]))
          #print('fill agnularity: ',              np.divide(angularity-fat_jets[i]["Angularity"],fat_jets[i]["Angularity"]))
          #print('fill aplanarity: ',                 np.divide(aplanarity-fat_jets[i]["Aplanarity"],fat_jets[i]["Aplanarity"]))
          #print('fill ktdr:',           np.divide(KtDR-fat_jets[i]["KtDR"],fat_jets[i]["KtDR"]))
          #print('fill planarflow:',            np.divide(planarFlow-fat_jets[i]["PlanarFlow"],fat_jets[i]["PlanarFlow"]))
          #print('fill Zcut12: ', np.divide(zcut-fat_jets[i]["ZCut12"],fat_jets[i]["ZCut12"]))
          #print('fill Qw: ',         np.divide(qw-fat_jets[i]["Qw"],fat_jets[i]["Qw"]))


          #push to respective groups (signal, validation, training)
          #hlvs_signal.append(tmp_hlvs)
        pass
 
        if len(hlvs_signal) >= 0: 
          sorted_hlvs_signal[str(n_c)].append(hlvs_signal) 

  #------------------------------------------------------------------------------------------------------------------------------------
  #------------------------------------------------------------------------------------------------------------------------------------
  #gStyle.SetOptStat(0)
  c1 = TCanvas('c1', 'c1', 500,400)
  c1.SetLogy()
  hist_c2.Draw("hist")
  c1.SaveAs("plots/hist_c2.pdf")
  hist_d2.Draw("hist")
  c1.SaveAs("plots/hist_d2.pdf")
  hist_tau1.Draw("hist")
  c1.SaveAs("plots/hist_tau1.pdf")
  hist_tau2.Draw("hist")
  c1.SaveAs("plots/hist_tau2.pdf")
  hist_tau3.Draw("hist")
  c1.SaveAs("plots/hist_tau3.pdf")
  hist_tau21.Draw("hist")
  c1.SaveAs("plots/hist_tau21.pdf")
  hist_tau32.Draw("hist")
  c1.SaveAs("plots/hist_tau32.pdf")
  hist_tau31.Draw("hist")
  c1.SaveAs("plots/hist_tau31.pdf")
  hist_split12.Draw("hist")
  c1.SaveAs("plots/hist_split12.pdf")
  hist_split23.Draw("hist")
  c1.SaveAs("plots/hist_split23.pdf")
  hist_angularity.Draw("hist")
  c1.SaveAs("plots/hist_angularity.pdf")
  hist_aplanarity.Draw("hist")
  c1.SaveAs("plots/hist_aplanarity.pdf")
  hist_ktdr.Draw("hist")
  c1.SaveAs("plots/hist_ktdr.pdf")
  hist_zcut.Draw("hist")
  c1.SaveAs("plots/hist_zcut.pdf")
  hist_qw.Draw("hist")
  c1.SaveAs("plots/hist_qw.pdf")
  hist_planarflow.Draw("hist")
  c1.SaveAs("plots/hist_planarflow.pdf")
  hists_file.Write()  
  hists_file.Close()

  #outfile_sig = h5py.File("events_anomalydetection_signal_boosted_VRNN.hdf5","w")
  #for i in range(n_start,n_consts):
  #  outfile_sig.create_dataset(str(i)+"/hlvs", data=sorted_hlvs_signal[str(i)])
  #outfile_sig.close()


