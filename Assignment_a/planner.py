import numpy as np
import pulp
import copy
import argparse

def vi(tt,rr,s,a,error,gamma):
  r=np.array(rr)
  t=np.array(tt)
  states= (s)
  acts= (a)
  v2= np.array([0.0 for i in range(states)])
  v= np.array([0.0 for i in range(states)])
  pi= np.array([0.0 for i in range(states)])
  while True:
    v = copy.deepcopy(v2)
    for st in range(0, states):
      max= float('-inf')
      maxi=0
      for j in range(0, acts):
        sum= float(np.dot((t[st][j]),(r[st][j]+ gamma*v)))
        #for i in range(0,states):
        #  sum= sum+ (t[st][j][i]*(r[st][j][i]+ gamma*v[i]))
        if (sum>max):
          max= sum
          maxi= j
      v2[st]= max
      pi[st]= maxi
    if (np.dot((v-v2),(v-v2)) < error):
      break
  return v, pi
  






      
def lp(t,r,s,a,error,gamma):
  states= (s)
  acts= (a)
  problem = pulp.LpProblem('task1', pulp.LpMaximize)
  array= []
  for i in range(0, states):
    array.append(pulp.LpVariable('a'+str(i)))
  
  sum=0.0
  for i in range(0, states):
    sum= sum+ array[i]
  sum= sum*-1
  problem += sum

  for i in range(0, acts):
    for j in range(0, states):
      sum=0.0
      for k in range(0, states):
        sum= sum+ t[j][i][k]*(r[j][i][k]+ (gamma*array[k]))
      problem += array[j] >= sum
  
  
  problem.solve(pulp.PULP_CBC_CMD(msg=False))

  v=[0 for x in range(states)]

  for i in problem.variables():
    #ind = int(i.name[1:]) if append not in the same order??
    v[int(i.name[1:])]= i.varValue



  pi=[]
  for st in range(0, states):
    max= float('-inf')
    maxi=0
    for j in range(0, acts):
      sum=0.0
      for i in range(0,states):
        sum= sum+ (t[st][j][i]*(r[st][j][i]+ gamma*v[i]))
      if (sum>max):
        max= sum
        maxi= j
      if (max==v[st]):
        break
    pi.append(maxi)
  
  return v, pi
         




    




def hi(tt,rr,s,a,error,gamma):

  r=np.array(rr)
  t=np.array(tt)
  states= s
  acts= a
  v=  np.array([0.0 for i in range(states)])
  pi2= np.array([0.0 for i in range(states)])
  v2=  np.array([0.0 for i in range(states)])
  
  pi= [0 for i in range(states)]
  sum=0.0


  while True:
    indic=0
    while True:
      v = copy.deepcopy(v2)
      for i in range(states):
        sum= float(np.dot((t[i][pi[i]]),(r[i][pi[i]] + (gamma*v))))
        #for j in range(states):
        #  sum= sum + t[i][pi[i]][j]*(r[i][pi[i]][j] + (gamma*v[j]))
        v2[i] = sum
      if (np.dot((v-v2),(v-v2)) < error):
        break


    pi2= copy.deepcopy(pi)
    for i in range(states):
      sum2 = float(np.dot((t[i][pi2[i]]),(r[i][pi2[i]]+ gamma*v)))
      for j in range(acts):
        #sum=0
        sum= float(np.dot((t[i][j]),(r[i][j]+ gamma*v)))
        #for k in range(states):
        #  sum= sum + t[i][j][k]*(r[i][j][k] + (gamma*v[k]))
        if (sum-sum2>1e-12):
          pi[i]= j
          indic=1
          break

    if indic==0:
      break

  return v, pi






    

parser = argparse.ArgumentParser()
parser.add_argument("--mdp")
parser.add_argument("--algorithm")
args = parser.parse_args()

filename = args.mdp
algor = args.algorithm

if algor==None:
  algor= "vi"

#with open("episodic-mdp-50-20.txt", "r") as f:
with open(filename, "r") as f:
  lines= f.readlines()
  k= len(lines)
  words = [lines[i][0:len(lines[i])].split() for i in range(0,k)]
discount=0
states= int(words[0][1])
acts= int(words[1][1]) 
t= [[[0 for x in range(states)] for y in range(acts)] for z in range(states)]
r= [[[0 for x in range(states)] for y in range(acts)] for z in range(states)]

for value in words:
    if value[0] == 'transition':
        i= int(value[1])
        j= int(value[2])
        k= int(value[3])
        t[i][j][k]= float(value[5])
        r[i][j][k]= float(value[4])
    elif value[0] == 'discount':
        discount = float(value[1])


error= 1e-20


#vo, pi = hi(t, r, states, acts, error, discount)

if (algor == 'vi'):
    vf, pi = hi(t, r, states, acts, error, discount)
elif (algor == 'lp'):
    vf, pi = lp(t, r, states, acts, error, discount)
elif (algor == 'hpi'):
    vf, pi = hi(t, r, states, acts, error, discount)


for i in range(len(vf)):
    print('{:.6f}'.format(round(vf[i], 6)) + "\t" + str(int(pi[i])))









    








