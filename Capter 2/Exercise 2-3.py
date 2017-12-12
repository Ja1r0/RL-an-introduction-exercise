import random

class bandit:
	def __init__(self,arm_num):
		self.arms=[]
		for _ in range(arm_num):
			arm={'mean':random.gauss(0,1),'std':1}
			self.arms.append(arm)
	def reset(self):
		for i in range(len(self.arms)):
			self.arms[i]['mean']=random.gauss(0,1)
	def step(self,arm_idx):
		mean=self.arms[arm_idx]['mean']
		std=self.arms[arm_idx]['std']
		reward=random.gauss(mean,std)
		return reward
class agent:
	def __init__(self,greedy,arm_num):
		self.greedy=greedy
		self.arm_num=arm_num
		self.memory=[]
		for _ in range(arm_num):
			arm_info={'Q':0.0,'visits':0}
			self.memory.append(arm_info)		
	def update_memory(self,arm_idx,reward):
		Q=self.memory[arm_idx]['Q']
		visits=self.memory[arm_idx]['visits']
		total_reward=Q*visits
		visits+=1
		self.memory[arm_idx]['Q']=toral_reward/visits
		self.memory[arm_idx]['visits']=visits		
	def action(self):
		prob=random.uniform(0,1)
		if prob<self.greedy:
			arm_idx=random.randrange(0,self.arm_num)
		else:
			arm_idx=self.memory.index(max(self.memory))
		return arm_idx
