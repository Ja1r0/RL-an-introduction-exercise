import random
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 对元组的列表进行排序时，默认依据元组中的第一个元素，若想依据第二个元素，可以写为：
# l=[(a,b),(),()]
# sortd(l,key=lambda x:x[1])
# max(l,key=lambda x:x[1])
# 这同样适用于元组为namedtuple时
# getattr(object,'attribute_name')函数可以获得一个对象的属性或方法
class env:
    def __init__(self, arms_num):
        self.arms = []
        bandit_arm = namedtuple('bandit_arm', ['mean', 'std'])
        for i in range(arms_num):
            arm = bandit_arm(random.gauss(0, 1), 1)
            self.arms.append(arm)
        self.optimal_arm_idx = self.arms.index(max(self.arms))

    def reset(self):
        pass

    def step(self, arm_idx):
        arm = self.arms[arm_idx]
        reward = random.gauss(arm.mean, arm.std)
        return reward


class agent:
    def __init__(self, greedy, arms_num):
        self.greedy = greedy
        self.arms_num=arms_num
        self.memory = []
        arm_info = namedtuple('arm_info', ['Q', 'visits'])
        for i in range(self.arms_num):
            arm = [0.0, 0]
            self.memory.append(arm)
        self.reward_list = []

    def update_memory(self, arm_idx, reward):
        Q = self.memory[arm_idx][0]
        visits = self.memory[arm_idx][1]
        total_reward = Q * visits + reward
        self.reward_list.append(reward)
        self.memory[arm_idx][0] = total_reward / (visits + 1)
        self.memory[arm_idx][1] += 1

    def action(self):
        prob = random.uniform(0, 1)
        if prob < self.greedy:
            arm_idx = random.randrange(0, self.arms_num)
        else:
            arm_idx = self.memory.index(max(self.memory))
        return arm_idx


def play(max_steps=1000, tasks_num=2000):
    tasks = []
    single_task = namedtuple('single_task', ['env', 'agent0', 'agent01', 'agent001'])
    for i in range(tasks_num):
        task = single_task(env(10), agent(0, 10), agent(0.1, 10), agent(0.01, 10))
        tasks.append(task)
    agent_data_contain=namedtuple('agent_data_contain',['average_reward','right_percent'])
    agents_data=[]
    agent_names=['agent0','agent01','agent001']
    for i in range(3):
        agent_name=agent_names[i]
        agent_right_percent = []        
        for step in range(1, max_steps + 1):
            if_optimal_arm = []           
            for task in tasks:
                ### agent ###
                arm_idx = getattr(task,agent_name).action()
                reward = task.env.step(arm_idx)
                getattr(task,agent_name).update_memory(arm_idx, reward)
                if arm_idx == task.env.optimal_arm_idx:
                    if_optimal_arm.append(1)
                else:
                    if_optimal_arm.append(0)             
                ######
            agent_right_percent.append(if_optimal_arm.count(1) / tasks_num            
        agent_reward_list = np.zeros(max_steps)     
        for task in tasks:
            agent_reward_list += np.array(getattr(task,agent_name).reward_list)         
        agent_average_reward = agent_reward_list / tasks_num
        agents_data.append(agent_data_contain(agent_average_reward,agent_right_percent))
    #figure 2.1
    plt.figure(21)
    sns.violinplot(data=np.random.randn(20000,10)+np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.show()
    #figure 2.2
    plt.figure(22)
    plt.subplot(2, 1, 1)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.plot(agents_data[1].average_reward, 'r', label='$\epsilon=0.1$')
    plt.plot(agents_data[2].average_reward, 'g', label='$\epsilon=0.01$')
    plt.plot(agents_data[0].average_reward, 'b', label='$\epsilon=0$(greedy)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel('Steps')
    plt.ylabel('Optimal action')
    plt.plot(agents_data[1].right_percent, 'r', label='$\epsilon=0.1$')
    plt.plot(agents_data[2].right_percent, 'g', label='$\epsilon=0.01$')
    plt.plot(agents_data[0].right_percent, 'b', label='$\epsilon=0$(greedy)')
    plt.legend()
    
    plt.show()


if __name__ == '__main__':
    play(max_steps=1000,
         tasks_num=2000)
