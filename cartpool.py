#This is implementation of CartPole-v1 in GYM enviroment

import gym
import numpy as np
import keras
import BasicAgent as Agent
import os

EPISODES = 400

if __name__ == '__main__':
	env = gym.make('CartPole-v1')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = Agent.CartpoolAgent_(state_size, action_size)

	done = False
	batchsize = 32

	# If you want to save a weight or load a weight make the variable true
	saveWeights = True
	loadWeights = False

	if loadWeights:
		cachedFolder = "savedweights/"
		agent.load(cachedFolder + "cartpole-dqn.h5")

	for episode in range(EPISODES):
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		for time in range(500):
			env.render()
			action = agent.play(state) # playing a move as per our present state
			next_state , reward , done , _ = env.step(action)
			reward = reward if not done else -10 # reward is -10 when we loose the game
			next_state = np.reshape(next_state, [1, state_size]) 
			agent.remember(state,action,reward,next_state,done)
			state = next_state
			if done:
				print ("Episode : {}/{} , Score : {} , epsilon : {:.2}".format(episode,EPISODES,time,agent.epsilon))
				break

		if len(agent.memory) > batchsize:
			agent.replayMemory(batchsize)

	if saveWeights:
		cachedFolder = "savedweights/"
		os.system('mkdir -p ' + cachedFolder)
		agent.save(cachedFolder + "cartpole-dqn.h5")

		