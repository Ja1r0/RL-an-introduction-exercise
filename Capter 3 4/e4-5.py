import numpy as np

def Sample_from_poisson(value_list,lamb):
    prob=np.zeros_like(value_list,dtype=np.float32)
    for i in range(len(value_list)):
        n=value_list[i]
        p=(lamb**n)/np.math.factorial(n)*np.exp(-1*lamb)
        prob[i]=p
    return prob
if __name__ == '__main__':
    value_list=np.arange(21)
    prob=Sample_from_poisson(value_list,lamb=4)
    print(prob)
    print(sum(prob))
class Policy_iteration:
    '''
    input
    =====
    act_space: {ndarray}
    obs_space: {ndarray}
    '''
    def __init__(self,act_space,obs_space):
        self.act_space=act_space
        self.obs_space=obs_space
        self.value=np.zeros_like(self.act_space)
        self.policy=None
    def evaluation(self,):
        pass

class Jack_car_rental:
    def __init__(self,):
        self.state=np.array([0.0,0.0])
        self.action_space=None
    def step(self,action):
        self.state[0]-=action
        self.state[1]+=action
        request=lamb**n/np.math.factorial(n)
        return reward,next_s
    def reset(self):
        pass
