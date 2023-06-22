'''
    1. Don't delete anything which is already there in code.
    2. you can create your helper functions to solve the task and call them.
    3. Don't change the name of already existing functions.
    4. Don't change the argument of any function.
    5. Don't import any other python modules.
    6. Find in-line function comments.

'''

import gym
import numpy as np
import math
import time
import argparse
import matplotlib.pyplot as plt

kv=10
kx=10
kx1=22
kv1=16

class sarsaAgent():
    '''
    - constructor: graded
    - Don't change the argument of constructor.
    - You need to initialize epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2 and weight_T1, weights_T2 for task-1 and task-2 respectively.
    - Use constant values for epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2.
    - You can add more instance variable if you feel like.
    - upper bound and lower bound are for the state (position, velocity).
    - Don't change the number of training and testing episodes.
    '''


    

    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.epsilon_T1 = 0.01
        self.epsilon_T2 = 0.01
        self.learning_rate_T1 = 0.04
        self.learning_rate_T2 = 0.18
        self.weights_T1 = [[[0 for j in range(kx)] for l in range(kv)] for x in range(3)]
        self.weights_T2 = [[[0 for j in range(kx1)] for l in range(kv1)] for x in range(3)]
        self.discount = 1.0
        self.train_num_episodes = 10000
        self.test_num_episodes = 100
        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]

    '''
    - get_table_features: Graded
    - Use this function to solve the Task-1
    - It should return representation of state.
    '''

    def get_table_features(self, obs):
        
        v= obs[1]
        x= obs[0]
        sv= int((v+0.07)*kv/0.14)
        sx= int((x+0.9+0.3)*kx/1.80)


        return [sx, sv]

    '''
    - get_better_features: Graded
    - Use this function to solve the Task-2
    - It should return representation of state.
    '''

    def get_better_features(self, obs):





        v= obs[1]
        x= obs[0]
        
        sv= ((v+0.07)*kv1/0.14)
        sx= ((x+0.3+0.9)*kx1/1.80)
        iv= int(sv)
        ix= int(sx)

        fv= sv- iv
        fx= sx- ix

        lessv=[0,0]#,0,0]
        morev=[0,0]#,0,0]
        lessx=[0,0]#,0,0]
        morex=[0,0]#,0,0]
        rx=[]
        rv=[]


        lessx[0]= ix-1
        lessx[1]= ix-2
        #lessx[2]= ix-3
        #lessx[3]= ix-4
        morex[0]= ix+1
        morex[1]= ix+2
        #morex[2]= ix+3
        #morex[3]= ix+4

        lessv[0]= iv-1
        lessv[1]= iv-2
        #lessv[2]= iv-3
        #lessv[3]= iv-4
        morev[0]= iv+1
        morev[1]= iv+2
        #morev[2]= iv+3
        #morev[3]= iv+4

        for i in range(len(lessx)):
            if lessx[i]<0:
                lessx[i]=0

        for i in range(len(morex)):
            if morex[i]>kx1-1:
                morex[i]=kx1-1

        for i in range(len(lessv)):
            if lessv[i]<0:
                lessv[i]=0

        for i in range(len(morev)):
            if morev[i]>kv1-1:
                morev[i]=kv1-1

        






        """
        if ix==0:
            lessx= ix
            morex= ix+1

        elif ix==19:
            lessx= ix-1
            morex= ix

        else:

            lessx= ix-1
            morex= ix+1


        if iv==0:
            lessv= iv
            morev= iv+1

        elif iv==19:
            lessv= iv-1
            morev= iv

        else:

            lessv= iv-1
            morev= iv+1


        if fv<=0.2:
            rv= [iv, iv, iv, lessv, lessv]
        elif fv<=0.4:
            rv= [iv, iv, iv, iv, lessv]
        elif fv<=0.6:
            rv= [iv, iv, iv, iv, iv]
        elif fv<=0.8:
            rv= [morev, iv, iv, iv, iv]
        else:
            rv= [morev, morev, iv, iv, iv]


        if fx<=0.2:
            rx= [ix, ix, ix, lessx, lessx]
        elif fx<=0.4:
            rx= [ix, ix, ix, ix, lessx]
        elif fx<=0.6:
            rx= [ix, ix, ix, ix, ix]
        elif fx<=0.8:
            rx= [morex, ix, ix, ix, ix]
        else:
            rx= [morex, morex, ix, ix, ix]

        """

        #rx=[lessx[3], lessx[2], lessx[1], lessx[0], ix, morex[0], morex[1], morex[2], morex[3]]
        #rv=[lessv[3], lessv[2], lessv[1], lessv[0] ,iv, morev[0], morev[1], morev[2], morev[3]]

        rx=[lessx[1], lessx[0], ix, morex[0], morex[1]]
        rv=[lessv[1], lessv[0] ,iv, morev[0], morev[1]]
        #if np.random.rand()<0.01:

        #   print(rx)

        return [rx, rv]




        #return None

    '''
    - choose_action: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function should return a valid action.
    - state representation, weights, epsilon are set according to the task. you need not worry about that.
    '''

    def choose_action(self, state, weights, epsilon):
        
        arr= np.array(state)
        arrsz= len(np.shape(arr))

        if arrsz==1:



            k= len(weights[0])
            

            maxa=0
            maxv= float('-inf')
            for i in range(3):
              if weights[i][state[1]][state[0]]>maxv:
                maxv= weights[i][state[1]][state[0]]
                maxa= i

            epp= np.random.rand()

            #print(maxa)   #deleteee

            if epp<epsilon:
              return int(np.random.rand()*3)

            else:
              return maxa


        else:

            
            sig= 1

            maxa=0
            maxv= float('-inf')
            for i in range(3):
                #sumaa=0
                sumaa1=  weights[i][state[1][2]][state[0][2]] + (  np.exp(-1/(2*sig*sig))* (weights[i][state[1][2]][state[0][1]] +  weights[i][state[1][1]][state[0][2]] +  weights[i][state[1][2]][state[0][3]] +  weights[i][state[1][3]][state[0][2]]))    +       (  np.exp(-2/(2*sig*sig))* (weights[i][state[1][3]][state[0][1]] +  weights[i][state[1][1]][state[0][3]] +  weights[i][state[1][3]][state[0][3]] +  weights[i][state[1][1]][state[0][1]]))   +       (  np.exp(-4/(2*sig*sig))* (weights[i][state[1][4]][state[0][2]] +  weights[i][state[1][2]][state[0][4]] +  weights[i][state[1][0]][state[0][2]] +  weights[i][state[1][2]][state[0][0]]))      +       (  np.exp(-5/(2*sig*sig))* (weights[i][state[1][4]][state[0][1]] +  weights[i][state[1][0]][state[0][1]] +  weights[i][state[1][4]][state[0][3]] +  weights[i][state[1][0]][state[0][3]]  +    weights[i][state[1][1]][state[0][4]] +  weights[i][state[1][1]][state[0][0]] +  weights[i][state[1][3]][state[0][4]] +  weights[i][state[1][3]][state[0][0]]                  )) 
                #for j in range(5):
                #    for l in range(5):
                #        for m in range(5):
                #            if (np.square(2-l)+np.square(2-m))==j:
                #                sumaa= sumaa + (  np.exp(-1*j/(2*sig*sig))* weights[i][state[1][l]][state[0][m]]  ) 

                if sumaa1>maxv:
                    maxa= i
                    maxv= sumaa1

            

            epp= np.random.rand()

            #print(maxa)   #deleteee

            if epp<epsilon:
              return int(np.random.rand()*3)

            else:
              return maxa





    '''
    - sarsa_update: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function will return the updated weights.
    - use sarsa(0) update as taught in class.
    - state representation, new state representation, weights, learning rate are set according to the task i.e. task-1 or task-2.
    '''

    def sarsa_update(self, state, action, reward, new_state, new_action, learning_rate, weights):
        
        arr= np.array(state)
        arrsz= len(np.shape(arr))

        if arrsz==1:


            k= len(weights[action])
            

            weights[action][state[1]][state[0]]= weights[action][state[1]][state[0]]*(1- learning_rate) + learning_rate*(reward + weights[new_action][new_state[1]][new_state[0]])
            

            return weights

        else:

            weights[action][state[1][2]][state[0][2]]= weights[action][state[1][2]][state[0][2]]*(1- learning_rate) + learning_rate*(reward + weights[new_action][new_state[1][2]][new_state[0][2]])
            #print(weights)

            return weights



    '''
    - train: Ungraded.
    - Don't change anything in this function.
    
    '''

    def train(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
            weights = self.weights_T1
            epsilon = self.epsilon_T1
            learning_rate = self.learning_rate_T1
        else:
            get_features = self.get_better_features
            weights = self.weights_T2
            epsilon = self.epsilon_T2
            learning_rate = self.learning_rate_T2
        reward_list = []
        plt.clf()
        plt.cla()
        for e in range(self.train_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            new_action = self.choose_action(current_state, weights, epsilon)
            while not done:
                action = new_action
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                new_action = self.choose_action(new_state, weights, epsilon)
                weights = self.sarsa_update(current_state, action, reward, new_state, new_action, learning_rate,
                                            weights)
                current_state = new_state
                if done:
                    reward_list.append(-t)
                    break
                t += 1
        self.save_data(task)
        reward_list=[np.mean(reward_list[i-100:i]) for i in range(100,len(reward_list))]
        plt.plot(reward_list)
        plt.savefig(task + '.jpg')

    '''
       - load_data: Ungraded.
       - Don't change anything in this function.
    '''

    def load_data(self, task):
        return np.load(task + '.npy')

    '''
       - save_data: Ungraded.
       - Don't change anything in this function.
    '''

    def save_data(self, task):
        if (task == 'T1'):
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T1)
            f.close()
        else:
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T2)
            f.close()

    '''
    - test: Ungraded.
    - Don't change anything in this function.
    '''

    def test(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
        else:
            get_features = self.get_better_features
        weights = self.load_data(task)
        reward_list = []
        for e in range(self.test_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            while not done:
                action = self.choose_action(current_state, weights, 0)
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                current_state = new_state
                if done:
                    reward_list.append(-1.0 * t)
                    break
                t += 1
        return float(np.mean(reward_list))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
       help="first operand", choices={"T1", "T2"})
    ap.add_argument("--train", required=True,
       help="second operand", choices={"0", "1"})
    args = vars(ap.parse_args())
    task=args['task']
    train=int(args['train'])
    agent = sarsaAgent()
    agent.env.seed(0)
    np.random.seed(0)
    agent.env.action_space.seed(0)
    if(train):
        agent.train(task)
    else:
        print(agent.test(task))