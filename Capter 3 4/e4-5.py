import numpy as np
import random
def Sample_from_poisson(value_list,lamb):
    prob=np.zeros_like(value_list,dtype=np.float32)
    for i in range(len(value_list)):
        n=value_list[i]
        p=(lamb**n)/np.math.factorial(n)*np.exp(-1*lamb)
        prob[i]=p
    m=np.random.multinomial(1,prob)
    num=value_list[np.nonzero(m)]
    return num
def Poisson_prob(n,lamb):
    return (lamb**n)/np.math.factorial(n)*np.exp(-1*lamb)



class Jack_car_rental:
    def __init__(self):
        self.state=np.array([0.0,0.0])
        self.state[0]=np.random.random_integers(0,20)
        self.state[1]=np.random.random_integers(0,20)
        self.action_space=np.arange(-5,5+1)
    def step(self,state,action):
        self.state=state
        self.state[0]-=action
        self.state[1]+=action
        requests_f=Sample_from_poisson(np.arange(1000),lamb=3)
        requests_s=Sample_from_poisson(np.arange(1000),lamb=4)
        returns_f=Sample_from_poisson(np.arange(1000),lamb=3)
        returns_s=Sample_from_poisson(np.arange(1000),lamb=2)
        self.state[0]=self.state[0]-requests_f+returns_f
        if self.state[0]>20:
            self.state[0]=20
        self.state[1]=self.state[1]-requests_s+returns_s
        if self.state[1]>20:
            self.state[1]=20
        reward=(requests_f+requests_s)*10-action*2
        return reward,self.state
    def reset(self):
        pass
### action space ###
action_space=np.arange(-5,5+1) # {ndarray}
### station space ###
single_state_space=np.arange(0,20+1)
state_space=[]
for i in single_state_space:
    for j in single_state_space:
        state_space.append((i,j)) # {list} of {tuple}
### possible number of requested cars ###
requests=np.arange(0,11+1)

def Transition_prob(state,action):
    trans_prob=[]
    for next_state in state_space:
        r_prob={}
        for request_f in requests:
            for request_s in requests:
                s_f=state[0]-action
                s_s=state[1]+action
                return_f=next_state[0]-s_f+request_f
                return_s=next_state[1]-s_s+request_s
                if not(s_f<0 or s_s<0 or return_f<0 or return_s<0):
                    p=Poisson_prob(request_f,3)*Poisson_prob(request_s,4)\
                      *Poisson_prob(return_f,3)*Poisson_prob(return_s,2)
                    reward = (request_f + request_s) * 10 - action * 2
                    if reward in r_prob.keys():
                        r_prob[reward]+=p
                    else:
                        r_prob[reward]=p
        trans_prob.append({next_state:r_prob})
    return trans_prob

if __name__ == '__main__':
    #state=random.choice(state_space)
    #action=np.random.choice(action_space)
    state=(0,9)
    action=0
    trans_prob=Transition_prob(state,action)
    #print(state)
    #print(action)
    #print(trans_prob)
    prob_list=[]
    for d in trans_prob:
        for key,item in d.items():
            for k,i in item.items():
                prob_list.append(i)
    max=max(prob_list)
    print(max)
    min=min(prob_list)
    print(min)
    print(sum(prob_list))
    for d in trans_prob:
        for key,item in d.items():
            for k,i in item.items():
                item[k]=(item[k]-min)/(max-min)
    l=[]
    for d in trans_prob:
        for key,item in d.items():
            for k,i in item.items():
                l.append(i)
    print(sum(l))






class Policy_iteration:
    '''
    input
    =====
    act_space: {ndarray}
    obs_space: {ndarray}
    '''
    def __init__(self,act_space,sts_space):
        self.act_space=act_space
        self.stat_space=sts_space
        zeros=np.zeros_like(self.act_space,dtype=np.float32)
        self.value=dict(zip(self.stat_space,zeros))
        state=[]
        action=[]
        for st in self.stat_space:
            ac=np.random.choice(self.act_space)
            state.append(st)
            action.append(ac)
        self.policy=dict(zip(state,action))
    def evaluation(self,theta):
        delta=0.0
        for state in self.stat_space:
            v=self.value[state]

            reward,next_state=env.step(state,self.policy[state])
            self.value[state]=None


        pass