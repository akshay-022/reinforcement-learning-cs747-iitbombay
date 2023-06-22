

import numpy as np1
import pulp
import copy

def vi(tt,rr,s,a,error,gamma):
  r=np1.array(rr)
  t=np1.array(tt)
  states= (s)
  acts= (a)
  v2= np1.array([0.0 for i in range(states)])
  v= np1.array([0.0 for i in range(states)])
  pi= np1.array([0.0 for i in range(states)])
  while True:
    v = copy.deepcopy(v2)
    for st in range(0, states):
      max= float('-inf')
      maxi=0
      for j in range(0, acts):
        sum= float(np1.dot((t[st][j]),(r[st][j]+ gamma*v)))
        #for i in range(0,states):
        #  sum= sum+ (t[st][j][i]*(r[st][j][i]+ gamma*v[i]))
        if (sum>max):
          max= sum
          maxi= j
      v2[st]= max
      pi[st]= maxi
    if (np1.dot((v-v2),(v-v2)) < error):
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
         






def checkwin(k):
  if k[0]==k[4] and k[4]==k[8] and int(k[4])!=0:
    return (3-int(k[0]))
  elif k[2]==k[4] and k[6]==k[2] and int(k[2])!=0:
    return (3-int(k[2]))
  elif k[0]==k[1] and k[1]==k[2] and int(k[2])!=0:
    return (3-int(k[1]))
  elif k[3]==k[4] and k[4]==k[5] and int(k[5])!=0:
    return (3-int(k[3]))
  elif k[6]==k[7] and k[7]==k[8] and int(k[6])!=0:
    return (3-int(k[6]))
  elif k[0]==k[3] and k[6]==k[3] and int(k[3])!=0:
    return (3-int(k[3]))
  elif k[1]==k[4] and k[7]==k[4] and int(k[4])!=0:
    return (3-int(k[1]))
  elif k[2]==k[5] and k[8]==k[2] and int(k[5])!=0:
    return (3-int(k[2]))
  elif (int(k[0])*int(k[1])*int(k[2])*int(k[3])*int(k[4])*int(k[5])*int(k[6])*int(k[7])*int(k[8]))!=0:
    return (5)

  else:
    return (0)
  




















































#parser = argparse.ArgumentParser()
#parser.add_argument("--policy")
#parser.add_argument("--states")
#args = parser.parse_args()


#statesfile = args.states
#ppolicy = args.policy

with open("data/attt/policies/p1_policy2.txt", 'r') as f:
    lines= f.readlines()
    kas= len(lines)
    wordsas = [lines[i] for i in range(0,kas)]
    finall1= [(lines[i][0:len(lines)].split())[1:] for i in range(1,kas)]

finall1= [finall1[i] for i in range(len(finall1))]
final1= finall1.copy()
final1_1= [[0.0 for x in range(0,len(final1[0]))] for y in range(0,len(final1))]

for i in range(len(finall1)):
  for j in range(len(finall1[0])):
    final1[i][j]= float(finall1[i][j])


with open("policy1.txt", 'w') as f:
    for i in range(0,kas):
      f.write(wordsas[i])


with open("data/attt/policies/p2_policy1.txt", 'r') as f:
    lines= f.readlines()
    kas= len(lines)
    wordsas = [lines[i] for i in range(0,kas)]
    finall2= [(lines[i][0:len(lines)-1].split())[1:] for i in range(1,kas)]

finall2= [finall2[i] for i in range(len(finall2))]
final2= finall2.copy()
final2_1= [[0.0 for x in range(0,len(final2[0]))] for y in range(0,len(final2))]


for i in range(len(finall2)):
  for j in range(len(finall2[0])):
    final2[i][j]= float(finall2[i][j])


with open("policy2.txt", 'w') as f:
    for i in range(0,kas):
      f.write(wordsas[i])



with open("error.txt", "w") as f:
  f.write('\n')


nplayer=1

for lm in range(20):

  """
  if lm==0:

    if nplayer==1:
      ppolicy= "policy1.txt"
    else:
      ppolicy= "policy2.txt"
  

  else:
  """

  if nplayer==1:
    ppolicy= "policy2.txt"
  else:
    ppolicy= "policy1.txt"

  
  with open(ppolicy, "r") as f:
    lines= f.readlines()
    k= len(lines)
    words = [lines[i][0:len(lines[i])].split() for i in range(0,k)]

  playerind=0 
  nplayer= int(words[0][0])


  statesfile= "data/attt/states/states_file_p2.txt"
  with open(statesfile, "r") as f:
    lines= f.readlines()
    k= len(lines)
    states2 = [lines[i][0:len(lines[i])-1] for i in range(0,k)]

  statesfile1= "data/attt/states/states_file_p1.txt"
  with open(statesfile1, "r") as f:
    lines= f.readlines()
    k= len(lines)
    states1 = [lines[i][0:len(lines[i])-1] for i in range(0,k)]




  acts=9

  states1dict= {}
  states2dict= {}

  for i in range(len(states1)):
    states1dict[states1[i]]= i

  for i in range(len(states2)):
    states2dict[states2[i]]= i

  states1len= len(states1)
  states2len= len(states2)

  t1=[[[0 for x in range(states1len+1)] for y in range(acts)] for z in range(states1len+1)]
  r1=[[[0 for x in range(states1len+1)] for y in range(acts)] for z in range(states1len+1)]

  t2=[[[0.0 for x in range(states2len+1)] for y in range(acts)] for z in range(states2len+1)]
  r2=[[[0.0 for x in range(states2len+1)] for y in range(acts)] for z in range(states2len+1)]








  if (nplayer==1):
    
    for i in range(states2len):
      
      st= states2[i]
      flag=0
      for a in range(0,9):
        if st[a]=='0':
          ns= st[0:a]+"2"+st[a+1:]
          indic1= checkwin(ns)
          if indic1==1:
            flag=1
            t2[i][a][states2len]=1          #loss
          elif indic1==5:
            flag=1
            t2[i][a][states2len]=1          #draw
          else: 
            for k in range(0,len(words)):
              if words[k][0]==ns:

                for j in range(0,9):
                  if (ns[j]=='0'):
                    sdash= ns[0:j]+"1"+ ns[j+1:]           #??
                    prob= float(words[k][j+1])
                    indic2=checkwin(sdash)
                    if indic2==2:
                      r2[i][a][states2len]= 1
                      t2[i][a][states2len]= prob 
                      flag=1                              #win
                    elif indic2==5:
                      t2[i][a][states2len]= prob 
                      flag=1                              #draw
                    else:       
                      nextstate= states2dict[sdash]
                      t2[i][a][nextstate]= prob







  if (nplayer==2):
    
    for i in range(states1len):
      
      st= states1[i]
      flag=0
      for a in range(0,9):
        if st[a]=='0':
          ns= st[0:a]+"1"+st[a+1:]
          indic1= checkwin(ns)
          if indic1==2:
            flag=1
            t1[i][a][states1len]=1          #loss
          elif indic1==5:
            flag=1
            t1[i][a][states1len]=1          #draw
          else: 
            for k in range(0,len(words)):
              if words[k][0]==ns:

                for j in range(0,9):
                  if (ns[j]=='0'):
                    sdash= words[k][0][0:j]+"2"+ words[k][0][j+1:]           #??
                    prob= float(words[k][j+1])
                    indic2=checkwin(sdash)
                    if indic2==1:
                      r1[i][a][states1len]= 1
                      t1[i][a][states1len]= prob 
                      flag=1                                      #win
                    elif indic2==5:
                      t1[i][a][states1len]= prob 
                      flag=1                                      #draw
                    else:       
                      nextstate= states1dict[sdash]
                      t1[i][a][nextstate]= prob










  if nplayer==1:
    lenofst= len(states2)
  else:
    lenofst= len(states1)
















  if nplayer==1:
    vo, pio= vi(t2,r2 ,(lenofst+1) ,9 , 1e-20, 1)
  else:
    vo, pio= vi(t1,r1 ,(lenofst+1) ,9 , 1e-20, 1)
















  words = [[vo[i],pio[i]] for i in range(0,len(vo))]

  statesdict={}
  states2dict={}

  np=3-nplayer

  if np==2:
    states= states2
  else:
    states= states1

  for i in range(len(states)):
    statesdict[i]= states[i]
    states2dict[states[i]]= i 

  if nplayer==1:
    writefile= 'policy2.txt'
    writefile1= 'policy2_1.txt'
    final2_1= final2.copy() 
    iterfile= "additionalfiles/"+"p"+str(3-nplayer)+"iter"+str(int((lm/2)+1))
  else:
    writefile= 'policy1.txt'
    writefile1= 'policy1_1.txt'
    final1_1= final1.copy()
    iterfile= "additionalfiles/"+"p"+str(3-nplayer)+"iter"+str(int((lm/2)+1))





  with open(writefile, 'r') as f:
      lines= f.readlines()
      kas= len(lines)
      wordsas = [lines[i] for i in range(0,kas)]


  with open(writefile1, 'w') as f:
      for i in range(0,kas):
        f.write(wordsas[i])





  with open(writefile, 'w') as f:
      f.write(str(np))
      f.write('\n')

  with open(iterfile, 'w') as f:
      f.write(str(np))
      f.write('\n')

  

  #just extract the name of the state somehow and do in string
  strline=""
  for i in range(len(words)-1):
    k=0
    flag=0
    st= statesdict[i]
    strline= st 
    arr=[]
    if st[int(words[i][1])]=='0':
      for j in range(9):
          if int(words[i][1])==j:
            strline= strline+ " 1"
            arr.append(1.0)
          else:
            strline= strline+ " 0"
            arr.append(0.0)

    else:
      for j in range(9):
        if st[j]=='0':
          k=k+1
      for j in range(9):
          if st[j]=='0':
            strline= strline+ " "+ str(float(1.0/k))
            arr.append(float(1.0/k))
          else:
            strline= strline+ " 0"
            arr.append(0.0)

    if nplayer==1:
      final2[i]=arr

    else:
      final1[i]=arr

    #iterfile= "p"+str(3-nplayer)+"iter"+str(int((lm/2)+1))
    if nplayer==1:
      with open(iterfile, 'a') as f:
        f.write(strline)
        f.write('\n')


    with open(writefile, 'a') as f:
      f.write(strline)
      f.write('\n')

  with open("error.txt", 'a') as f:
    k1 = np1.linalg.norm(np1.subtract(np1.array(final1), np1.array(final1_1)))
    k2= np1.linalg.norm(np1.subtract(np1.array(final2), np1.array(final2_1)))
    f.write("agent 1 "+str(k1)+" agent 2 "+ str(k2))
    f.write('\n')




      

    


