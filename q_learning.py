# -*- coding: utf-8 -*-

import numpy as np
import random

class Q_learning(object):
    def __init__(self):        
        R = np.array([[-1,-1,-1,-1,0,-1],
              [-1,-1,-1,0,-1,100],
              [-1,-1,-1,0,-1,-1],
              [-1,0,0,-1,0,-1],
              [0,-1,-1,0,-1,100],
              [-1,0,-1,-1,0,100]],'float64')
        Q = np.zeros(R.shape,R.dtype)
        aim = 5
        self.load(R,aim,Q)
        
    def load(self,R,aim,Q=''):
        if(Q==''):
            Q=np.zeros(R.shape,R.dtype)
        self.states = Q.shape[0]
        assert(Q.shape[1]==self.states)
        assert(Q.shape==R.shape)
        self.Q = Q
        self.R = R
        self.AIM = aim

    def get_random_state(self):
        return random.randint(0,self.states-1)
        
    def get_next_state(self,state):
        state = self.R[state,:] != -1
        count = int(state.sum())
        state = (state*np.arange(self.states))
        state.sort()
        return state[random.randint(-count,-1)]    
        
    def get_best_state(self,state):
        state = [(self.Q[state,i],i) for i in range(self.states)]     
        state.sort(reverse=True)
        return state[0][1]
        
        
    def train(self,gamma=0.8,episode=100):
        Q,R =self.Q,self.R
        for i in range(episode):
            state = self.get_random_state()
            count = self.states**2
            while(state!=self.AIM):
                ss = self.get_next_state(state)
                Q[state,ss] = R[state,ss] + gamma*Q[ss,:].max()
                state = ss
                count -= 1
        return Q
        
        
def Hanoi_able(s1,s2):#步骤是否可行
        change = sum((s1[i]!=s2[i] for i in range(3)))
        if change != 1:
            return False#只能移动一块砖
        for change in range(3):
            if(s1[change]!=s2[change]):
                break
        for upper in range(change+1,3):
            if s1[upper]==s1[change]:
                return False#这块砖上不能有小的
            if s2[upper]==s2[change]:
                return False#移动到的位置下面不能有小的
        return True
        
def Hanoi():      
    state = []
    for i in range(3):
        for j in range(3):
            for g in range(3):
                state.append((i,j,g))
    aim = (2,2,2)
    states = len(state)
    R = np.ones((states,states),'float64')*-1
    for i in range(states):
        for j in range(states):
            if(Hanoi_able(state[i],state[j])):
                if(state[j]==aim):
                    R[i,j]=100
                else:
                    R[i,j]=0
            
    aim = state.index(aim)
    return (R,aim,state)

def Hanoi_shape(s):
    global buffer,ss
    ss = np.array([[1,1,1,1,1],
                   [0,1,1,1,0],
                   [0,0,1,0,0]],'int32');
    buffer = np.zeros((3,15),'int32')
    tower = [[],[],[]]
    for i in range(3):
        tower[s[i]].append(i)
    for i in range(3):
        for j,g in enumerate(tower[i]):
            buffer[2-j,i*5:i*5+5]=ss[g,:]
    return buffer
def Hanoi_print(s1,s2):#打印状态转移图案
    buffer = np.zeros((3,30),'int32')
    buffer[:,:15]=Hanoi_shape(s1)
    buffer[:,15:]=Hanoi_shape(s2)
    for i in range(3):
        for j in range(30):
            if j%5==0:
                print('|',end='')
            if j == 15:
                if i == 1:
                    print(' -> |',end='')
                else:
                    print('    |',end='')
            if buffer[i,j]==1:
                print('-',end='')
            else:
                print(' ',end='')
        print('|')
    
def main():
    R,aim,state = Hanoi()
    q = Q_learning()
    q.load(R,aim)
    Q = q.train()
    Q = np.int32(Q/Q.max()*100)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            print("%4d"%Q[i,j],end='')
        print()
    s = 0
    no = 0
    while(s!=aim):
        no+=1
        print("No.",no)
        ss = q.get_best_state(s)
        Hanoi_print(state[s],state[ss])
        s = ss
    print("%d moves!"%no)
    
    while(1):
        s = input("your state(a,b,c): ")
        s = s.split(',')
        s = tuple((int(i) for i in s))
        s = state.index(s)
        no=0
        while(s!=aim):
            no+=1
            print("No.",no)
            ss = q.get_best_state(s)
            Hanoi_print(state[s],state[ss])
            s = ss
if __name__ == '__main__':
    main()