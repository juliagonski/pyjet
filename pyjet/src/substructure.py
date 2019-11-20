import numpy as np



def deltaR(const1, const2):
  return np.square(const1.y - const2.y) + np.square(const1.phi - const2.phi)


def ECF1(jet):
  c = jet.constituents()
  n_consts = len(c)
  ecf1 = 0
  for i in range(n_consts):
    ecf1 += c[i].pt()
  return ecf1

def ECF2(jet, beta=2):
  c = jet.constituents()
  n_consts = len(c)
  ecf2 = 0
  for i in range(n_consts-1):
    for j in range(i+1, n_consts):
      ecf2 += c[i].pt()*c[j].pt()*np.pow(deltaR(c[i], c[j]), beta)
  return ecf2

def ECF3(jet, beta=2):
  c = jet.constituents()
  n_consts = len(c)
  ecf3 = 0
  for i in range(n_consts-2):
    for j in range(i+1, n_consts-1):
      for k in range(j+1, n_consts):
        ecf3 += c[i].pt()*c[j].pt()*c[k].pt()*np.pow(deltaR(c[i], c[j])*deltaR(c[i], c[k])*deltaR(c[j], c[k]), beta)
  return ecf3

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


def C2(jet, ecf_list = None):
  if ecf_list is None: ecf_list = ECF(jet)
  return ((ecf[2]*ecf[0])/np.square(ecf[1]))



def D2(jet, ecf_list = None):
  if ecf_list is None: ecf_list = ECF(jet)
  return ((ecf[2]*np.pow(ecf[0], 3))/np.pow(ecf[1], 3))
