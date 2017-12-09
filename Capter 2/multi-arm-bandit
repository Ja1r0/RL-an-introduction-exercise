import random
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt


# 对元组的列表进行排序时，默认依据元组中的第一个元素，若想依据第二个元素，可以写为：
# l=[(a,b),(),()]
# sortd(l,key=lambda x:x[1])
# max(l,key=lambda x:x[1])
# 这同样适用于元组为namedtuple时
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
    agent0_right_percent = []
    agent01_right_percent = []
    agent001_right_percent = []
    for step in range(1, max_steps + 1):
        if_optimal_arm_agent0 = []
        if_optimal_arm_agent01 = []
        if_optimal_arm_agent001 = []
        for task in tasks:
            ### agent0 ###
            arm_idx = task.agent0.action()
            reward = task.env.step(arm_idx)
            task.agent0.update_memory(arm_idx, reward)
            if arm_idx == task.env.optimal_arm_idx:
                if_optimal_arm_agent0.append(1)
            else:
                if_optimal_arm_agent0.append(0)
            ### agent01 ###
            arm_idx = task.agent01.action()
            reward = task.env.step(arm_idx)
            task.agent01.update_memory(arm_idx, reward)
            if arm_idx == task.env.optimal_arm_idx:
                if_optimal_arm_agent01.append(1)
            else:
                if_optimal_arm_agent01.append(0)
            ### agent001 ###
            arm_idx = task.agent001.action()
            reward = task.env.step(arm_idx)
            task.agent001.update_memory(arm_idx, reward)
            if arm_idx == task.env.optimal_arm_idx:
                if_optimal_arm_agent001.append(1)
            else:
                if_optimal_arm_agent001.append(0)
                ######
        agent0_right_percent.append(if_optimal_arm_agent0.count(1) / tasks_num)
        agent01_right_percent.append(if_optimal_arm_agent01.count(1) / tasks_num)
        agent001_right_percent.append(if_optimal_arm_agent001.count(1) / tasks_num)
    agent0_reward_list = np.zeros(max_steps)
    agent01_reward_list = np.zeros(max_steps)
    agent001_reward_list = np.zeros(max_steps)
    for task in tasks:
        agent0_reward_list += np.array(task.agent0.reward_list)
        agent01_reward_list += np.array(task.agent01.reward_list)
        agent001_reward_list += np.array(task.agent001.reward_list)
    agent0_average_reward = agent0_reward_list / tasks_num
    agent01_average_reward = agent01_reward_list / tasks_num
    agent001_average_reward = agent001_reward_list / tasks_num
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.plot(agent01_average_reward, 'r', label='$\epsilon=0.1$')
    plt.plot(agent001_average_reward, 'g', label='$\epsilon=0.01$')
    plt.plot(agent0_average_reward, 'b', label='$\epsilon=0$(greedy)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel('Steps')
    plt.ylabel('Optimal action')
    plt.plot(agent01_right_percent, 'r', label='$\epsilon=0.1$')
    plt.plot(agent001_right_percent, 'g', label='$\epsilon=0.01$')
    plt.plot(agent0_right_percent, 'b', label='$\epsilon=0$(greedy)')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    play(max_steps=1000,
         tasks_num=2000)
