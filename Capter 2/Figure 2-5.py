import random
import numpy as np
import math
from collections import namedtuple
import matplotlib.pyplot as plt


class bandits:
    def __init__(self, mean):
        self.arm_num = 10
        self.arms = []
        self.mean = mean
        for i in range(self.arm_num):
            mean = random.gauss(self.mean, 1)
            arm = {'mean': mean, 'std': 1}
            self.arms.append(arm)
        self.optimal_idx = self.arms.index(max(self.arms, key=lambda x: x['mean']))

    def step(self, arm_idx):
        arm = self.arms[arm_idx]
        mean = arm['mean']
        std = arm['std']
        reward = random.gauss(mean, std)
        return reward

    def reset(self):
        pass


class gradient_algo:
    def __init__(self, alpha, if_baseline):
        self.alpha = alpha
        self.if_baseline = if_baseline
        self.memory = []
        self.reward_list = []
        self.total_reward = 0.0
        self.arm_num = 10
        for i in range(self.arm_num):
            P = 1.0 / self.arm_num
            arm = {'H': 0.0, 'P': P, 'visits': 0}
            self.memory.append(arm)

    def update(self, arm_idx, reward, time_step):
        self.total_reward += reward
        if self.if_baseline:
            average_reward = self.total_reward / time_step
        else:
            average_reward = 0.0

        for arm in self.memory:
            if self.memory.index(arm) == arm_idx:

                arm['visits'] += 1
                arm['H'] += self.alpha * (reward - average_reward) * (1 - arm['P'])
            else:
                arm['H'] -= self.alpha*(reward - average_reward) * arm['P']
        Z = 0.0
        for arm in self.memory:
            Z += math.exp(arm['H'])
        for arm in self.memory:
            arm['P'] = math.exp(arm['H']) / Z

    def action(self):
        start = 0.0
        prob_sections = []
        for i in range(self.arm_num):
            arm = self.memory[i]
            end = start + arm['P']
            prob_sections.append({'start': start, 'end': end})
            start = end
        prob = random.random()
        for idx, section in enumerate(prob_sections):
            if section['start'] <= prob < section['end']:
                return idx


'''				
def play(task,max_step):
	for i in range(1,5):
		player_name='player'+'%d'%i
		player=getattr(task,player_name)
		if_optimal=[]
		for step in range(max_step):

			arm_idx=player.action()
			reward=task.env(arm_idx)
			player.update(arm_idx,reward,step)
			if arm_idx==env.optimal_idx:
				if_optimal.append(1)
			else:
				if_optimal.append(0)
'''


def figure_2_5(max_step=1000, task_num=2000):
    tasks = []
    single_task = namedtuple('single_task', ['env', 'player1', 'player2', 'player3', 'player4'])
    for _ in range(task_num):
        env = bandits(4)
        player1 = gradient_algo(0.1, True)
        player2 = gradient_algo(0.4, True)
        player3 = gradient_algo(0.1, False)
        player4 = gradient_algo(0.4, False)
        task = single_task(env, player1, player2, player3, player4)
        tasks.append(task)
    graph_data = []
    for i in range(1, 5):
        player_name = 'player' + '%d' % i
        optidx_percent = []
        for step in range(1,max_step+1):
            if_optimal = []
            for task in tasks:
                player = getattr(task, player_name)
                arm_idx = player.action()
                reward = task.env.step(arm_idx)
                player.update(arm_idx, reward, step)
                if arm_idx == task.env.optimal_idx:
                    if_optimal.append(1)
                else:
                    if_optimal.append(0)

            optidx_percent.append(if_optimal.count(1) / float(task_num))

        graph_data.append(optidx_percent)

    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('Optimal action')
    plt.plot(graph_data[0], 'r', label=r'$\alpha=0.1$,baseline')
    plt.plot(graph_data[1], 'g', label=r'$\alpha=0.4$,baseline')
    plt.plot(graph_data[2], 'b', label=r'$\alpha=0.1$,no baseline')
    plt.plot(graph_data[3], 'y', label=r'$\alpha=0.4$,no baseline')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    figure_2_5(max_step=1000, task_num=2000)



