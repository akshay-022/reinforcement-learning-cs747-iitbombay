


import argparse











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
  











parser = argparse.ArgumentParser()
parser.add_argument("--policy")
parser.add_argument("--states")
args = parser.parse_args()


statesfile = args.states
ppolicy = args.policy











#ppolicy= "p1_policy2.txt"
with open(ppolicy, "r") as f:
  lines= f.readlines()
  k= len(lines)
  words = [lines[i][0:len(lines[i])].split() for i in range(0,k)]

playerind=0 
nplayer= int(words[0][0])

if (nplayer==1):
  states1 = [words[i][0] for i in range(1,len(words))]
  playerind=2

if (nplayer==2):
  states2 = [words[i][0] for i in range(1,len(words))]
  playerind=1


#statesfile= "states_file_p2.txt"
with open(statesfile, "r") as f:
  lines= f.readlines()
  k= len(lines)
  if playerind==2:
    states2 = [lines[i][0:len(lines[i])-1] for i in range(0,k)]
  else:
    states1=  [lines[i][0:len(lines[i])-1] for i in range(0,k)]



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


  print('numStates '+ str(states2len+1))
  print('Actions '+ str(9))
  for i in range(states2len+1):
    for a in range(9):
      for k in range(states2len+1):
        if r2[i][a][k]!=0 or t2[i][a][k]!=0:
          print('transition '+str(i)+' '+ str(a)+' ' +str(k)+' '+str(r2[i][a][k])+' '+str(t2[i][a][k]))
          
        #if t2[i][a][k]!=0:
        #  print('transition '+str(i)+' '+ str(a)+' ' +str(k)+' '+str(r2[i][a][k])+' '+str(t2[i][a][k]))
  print('mdptype episodic')
  print('discount '+ str(1))




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


    print('numStates '+ str(states1len+1))
    print('Actions '+ str(9))
    for i in range(states1len):
      for a in range(9):
        for k in range(states1len):
          if r1[i][a][k]!=0 or t1[i][a][k]!=0:
            print('transition '+str(i)+' '+ str(a)+' ' +str(k)+' '+str(r1[i][a][k])+' '+str(t1[i][a][k]))
    print('mdptype episodic')
    print('discount '+ str(1))






