
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--value-policy")
parser.add_argument("--states")
parser.add_argument("--player-id")
args = parser.parse_args()

filename = args.value_policy
statesfile = args.states
playeri= args.player_id


with open(filename, "r") as f:
  lines= f.readlines()
  k= len(lines)
  words = [lines[i][0:len(lines[i])].split() for i in range(0,k)]

np= int(playeri)

statesdict={}
states2dict={}

with open(statesfile, "r") as f:
  lines= f.readlines()
  k= len(lines)
  states = [lines[i][0:len(lines[i])-1] for i in range(0,k)]

for i in range(len(states)):
  statesdict[i]= states[i]
  states2dict[states[i]]= i



print(str(np))


#just extract the name of the state somehow and do in string
strline=""
for i in range(len(words)-1):
  k=0
  flag=0
  st= statesdict[i]
  strline= st 
  
  if st[int(words[i][1])]=='0':
    for j in range(9):
        if int(words[i][1])==j:
          strline= strline+ " 1"
        else:
          strline= strline+ " 0"

  else:
    for j in range(9):
      if st[j]=='0':
        k=k+1
    for j in range(9):
        if st[j]=='0':
          strline= strline+ " "+ str(float(1.0/k))
        else:
          strline= strline+ " 0"


    

  print(strline)


