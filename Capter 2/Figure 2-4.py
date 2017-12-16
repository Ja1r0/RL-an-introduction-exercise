import random
import matplotlib.pyplot as plt
import numpy as np


class bandit:
    def __init__(self, arm_num):
        self.arms = []
        for _ in range(arm_num):
            arm = {'mean': random.gauss(0, 1), 'std': 1}
            self.arms.append(arm)

    def reset(self, random_walk_std):
        for i in range(len(self.arms)):
            self.arms[i]['mean'] += random.gauss(0, random_walk_std)

    def step(self, arm_idx):
        mean = self.arms[arm_idx]['mean']
        std = self.arms[arm_idx]['std']
        reward = random.gauss(mean, std)
        return reward

    def optim_idx(self):
        item = max(self.arms, key=lambda x: x['mean'])
        return self.arms.index(item)


class agent:
    def __init__(self, greedy, arm_num, alpha_type):
        self.greedy = greedy
        self.arm_num = arm_num
        self.memory = []
        for _ in range(arm_num):
            arm_info = {'Q': 0.0, 'visits': 0}
            self.memory.append(arm_info)
        self.alpha_type = alpha_type
        self.reward_list = []

    def update_memory(self, arm_idx, reward):
        Q = self.memory[arm_idx]['Q']
        visits = self.memory[arm_idx]['visits'] + 1
        alpha = 0.0
        if self.alpha_type == 'sample-average':
            alpha = 1.0 / visits
        elif self.alpha_type == 'constant':
            alpha = 0.1
        Q_new = Q + alpha * (reward - Q)
        self.memory[arm_idx]['Q'] = Q_new
        self.memory[arm_idx]['visits'] = visits
        self.reward_list.append(reward)

    def action(self, time_step):
        prob = random.uniform(0, 1)
        if prob < self.greedy:
            arm_idx = random.randrange(0, self.arm_num)
        else:
            item = max(self.memory, key=lambda x: x['Q'])
            arm_idx = self.memory.index(item)
        return arm_idx


class agent_UCB(agent):
    def __init__(self, c):
        self.c = c
        super(agent_UCB, self).__init__(greedy=None, arm_num=10, alpha_type='sample-average')

    def action(self, time_step):
        value = []
        for dict in self.memory:
            if dict['visits'] == 0:
                return self.memory.index(dict)
            confidence = self.c * np.sqrt(np.log(time_step) / dict['visits'])
            value.append(dict['Q'] + confidence)
        arm_idx = value.index(max(value))
        return arm_idx


def figure2_4(max_steps=1000, task_num=2000, random_walk_std=0.5):
    tasks = []
    for _ in range(task_num):
        task = bandit(10)
        agent0 = agent(0.1, 10, 'sample-average')
        agent1 = agent_UCB(c=2)
        tasks.append([task, agent0, agent1])
    graph1 = []
    graph2 = []
    for i in range(1, 3):
        optim_percent = []
        total_reward = np.zeros(max_steps)
        for step in range(1, max_steps + 1):
            optim_count = 0
            for single_task in tasks:
                player = single_task[i]
                env = single_task[0]
                arm_idx = player.action(step)
                reward = env.step(arm_idx)
                player.update_memory(arm_idx, reward)
                optim_idx = env.optim_idx()
                if arm_idx == optim_idx:
                    optim_count += 1
                    # env.reset(random_walk_std)
            optim_percent.append(optim_count / task_num)
        for single_task in tasks:
            player = single_task[i]
            total_reward += np.array(player.reward_list)
        average_reward = total_reward / task_num
        graph1.append(average_reward)
        graph2.append(optim_percent)
    plt.figure()
    # plt.subplot(2, 1, 1)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.plot(graph1[1], 'g', label=r'$UCB,c=2$')
    plt.plot(graph1[0], 'r', label=r'$\epsilon-greedy,\epsilon=0.1$')
    plt.legend()
    '''
	plt.subplot(2, 1, 2)
    plt.xlabel('Steps')
    plt.ylabel('Optimal action')
    plt.plot(graph2[0], 'r', label=r'$\alpha=\frac{1}{n}$')
    plt.plot(graph2[1], 'g', label=r'$\alpha=0.1$')
    plt.legend()
	'''
    plt.show()


if __name__ == '__main__':
    # random walk : q=q+e , e~N(0,std)
    figure2_4(max_steps=1000, task_num=2000, random_walk_std=0.5)