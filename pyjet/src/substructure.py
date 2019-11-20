import numpy as np



def deltaR(const1, const2):
  return np.square(const1.y - const2.y) + np.square(const1.phi - const2.phi)

#----------------------------------------------------------
# Energy Correlation Function 1
# sum_i(pt_i)
#
# Inputs: jet (Fastjet PseudoJet with constituents)
# Outputs: ECF1
#
def ECF1(jet):
  c = jet.constituents()
  n_consts = len(c)
  ecf1 = 0
  for i in range(n_consts):
    ecf1 += c[i].pt()
  return ecf1

#----------------------------------------------------------
# Energy Correlation Function (2, beta)
# sum_i,j (pt_i*pt_j*(deltaR_ij)^beta)
#     i<j
#
# Inputs: 
#         jet (Fastjet PseudoJet with constituents)
#         beta (default: 2)
#
#
# Outputs: ECF(2, beta)
#
def ECF2(jet, beta=2):
  c = jet.constituents()
  n_consts = len(c)
  ecf2 = 0
  for i in range(n_consts-1):
    for j in range(i+1, n_consts):
      ecf2 += c[i].pt()*c[j].pt()*np.pow(deltaR(c[i], c[j]), beta)
  return ecf2

#----------------------------------------------------------
# Energy Correlation Function (3, beta)
# sum_i,j,k (pt_i*pt_j*pt_k*(deltaR_ij*deltaR_ik*deltaR_jk)^beta)
#     i<j<k
#
# Inputs: 
#         jet (Fastjet PseudoJet with constituents)
#         beta (default: 2)
#
#
# Outputs: ECF(3, beta)
#
def ECF3(jet, beta=2):
  c = jet.constituents()
  n_consts = len(c)
  ecf3 = 0
  for i in range(n_consts-2):
    for j in range(i+1, n_consts-1):
      for k in range(j+1, n_consts):
        ecf3 += c[i].pt()*c[j].pt()*c[k].pt()*np.pow(deltaR(c[i], c[j])*deltaR(c[i], c[k])*deltaR(c[j], c[k]), beta)
  return ecf3

#----------------------------------------------------------
# Energy Correlation Function (4, beta)
# sum_i,j,k,l (pt_i*pt_j*pt_k*pt_l*(deltaR_ij*deltaR_ik*deltaR_il*deltaR_jk*deltaR_jl*deltaR_kl)^beta)
#     i<j<k<l
#
# Inputs: 
#         jet (Fastjet PseudoJet with constituents)
#         beta (default: 2)
#
#
# Outputs: ECF(4, beta)
#
def ECF4(jet, beta=2):
  c = jet.constituents()
  n_consts = len(c)
  ecf4 = 0
  for i in range(n_consts-3):
    for j in range(i+1, n_consts-2):
      for k in range(j+1, n_consts-1):
        for j in range(k+1, n_consts-1):
          ecf4 += c[i].pt()*c[j].pt()*c[k].pt()*c[l].pt()*np.pow(deltaR(c[i], c[j])*deltaR(c[i], c[k])*deltaR(c[i], c[l])*deltaR(c[j], c[k])*deltaR(c[j], c[l])*deltaR(c[k], c[l]), beta)
  return ecf4

#----------------------------------------------------------
# Energy Correlation Function List
#
# Inputs: 
#         jet (Fastjet PseudoJet with constituents)
#         List of desired ECF numbers (default: [1, 2, 3])
#         beta (default: 2)
#
#
# Outputs: List of ECFs for given input list
#
def ECF(jet, n_list=[1, 2, 3], beta=2):
  c = jet.constituents()
  n_consts = len(c)
  ecf1 = 0
  ecf2 = 0
  ecf3 = 0
  ecf4 = 0
  for i in range(n_consts):
    if (1 in n_list): ecf1 += c[i].pt()
    for j in range(i+1, n_consts-2):
      if (2 in n_list and i < n_consts-1): ecf2 += c[i].pt()*c[j].pt()*np.pow(deltaR(c[i], c[j]), beta)
      for k in range(j+1, n_consts-1):
        if (3 in n_list and i < n_consts-2 and j < n_consts-1): ecf3 += c[i].pt()*c[j].pt()*c[k].pt()*np.pow(deltaR(c[i], c[j])*deltaR(c[i], c[k])*deltaR(c[j], c[k]), beta)
        for j in range(k+1, n_consts-1):
          if (4 in n_list and i < n_consts-3 and j < n_consts-2 and k < n_consts-1): ecf4 += c[i].pt()*c[j].pt()*c[k].pt()*c[l].pt()*np.pow(deltaR(c[i], c[j])*deltaR(c[i], c[k])*deltaR(c[i], c[l])*deltaR(c[j], c[k])*deltaR(c[j], c[l])*deltaR(c[k], c[l]), beta)
  ecf_list = []
  if(1 in n_list): ecf_list.append(ecf1)
  if(2 in n_list): ecf_list.append(ecf2)
  if(3 in n_list): ecf_list.append(ecf3)
  if(4 in n_list): ecf_list.append(ecf4)
  return ecf_list 


#----------------------------------------------------------
# C2 Substructure Variable 
# ECF1*ECF3/ECF2^2
#
# Inputs: 
#         jet (Fastjet PseudoJet with constituents)
#         List of [ECF1, ECF2, ECF3] (default: None, will be determined by calling the ECF function defined above
#
#
# Outputs: C2
#
def C2(jet, ecf_list = None):
  if ecf_list is None: ecf_list = ECF(jet)
  return ((ecf[2]*ecf[0])/np.square(ecf[1]))

#----------------------------------------------------------
# D2 Substructure Variable 
# ECF1^3*ECF3/ECF2^3
#
# Inputs: 
#         jet (Fastjet PseudoJet with constituents)
#         List of [ECF1, ECF2, ECF3] (default: None, will be determined by calling the ECF function defined above
#
#
# Outputs: D2
#
def D2(jet, ecf_list = None):
  if ecf_list is None: ecf_list = ECF(jet)
  return ((ecf[2]*np.pow(ecf[0], 3))/np.pow(ecf[1], 3))

#----------------------------------------------------------
# 1-Subjettiness Substructure Variable 
# (1/d0)*sum_k(pt_k*min(deltaR_k,subjets))
# d0 = sum_k(pt_k*R0)
# R0 = Jet Radius
# Subjets found by unpacking clustering algorithm until there are N subjets (1 in this case)
#
# Inputs: 
#         jet (Fastjet PseudoJet with constituents)
#         jet radius (default: 1)
#
#
# Outputs: Tau 1
#
def Tau1(jet, radius=1):
  c = jet.constituents()
  n_consts = len(c)
  d0 = jet.pt()*radius
  d0tau = 0
  for i in range(n_consts):
    d0tau += c[i].pt()*deltaR(jet, c[i])
  return (d0tau/d0)
  
  
#----------------------------------------------------------
# 2-Subjettiness Substructure Variable 
# (1/d0)*sum_k(pt_k*min(deltaR_k,subjets))
# d0 = sum_k(pt_k*R0)
# R0 = Jet Radius
# Subjets found by unpacking clustering algorithm until there are N subjets (2 in this case)
#
# Inputs: 
#         jet (Fastjet PseudoJet with constituents)
#         jet radius (default: 1)
#
#
# Outputs: Tau 2
#
def Tau2(jet, radius=1):
  c = jet.constituents()
  n_consts = len(c)
  subjets = [] #TODO: Get subjets by unpacking clustering algorithm until there are 2 subjets 
  d0 = jet.pt()*radius
  d0tau = 0
  for i in range(n_consts):
    dRs = []
    for s in range(len(subjets)):
      dRs.append(deltaR(c[i], subjet[s]))
    d0tau += c[i].pt()*np.min(dRs)
  return (d0tau/d0)

#----------------------------------------------------------
# 3-Subjettiness Substructure Variable 
# (1/d0)*sum_k(pt_k*min(deltaR_k,subjets))
# d0 = sum_k(pt_k*R0)
# R0 = Jet Radius
# Subjets found by unpacking clustering algorithm until there are N subjets (3 in this case)
#
# Inputs: 
#         jet (Fastjet PseudoJet with constituents)
#         jet radius (default: 1)
#
#
# Outputs: Tau 3
#
def Tau3(jet, radius=1):
  c = jet.constituents()
  n_consts = len(c)
  subjets = [] #TODO: Get subjets by unpacking clustering algorithm until there are 3 subjets 
  d0 = jet.pt()*radius
  d0tau = 0
  for i in range(n_consts):
    dRs = []
    for s in range(len(subjets)):
      dRs.append(deltaR(c[i], subjet[s]))
    d0tau += c[i].pt()*np.min(dRs)
  return (d0tau/d0)

