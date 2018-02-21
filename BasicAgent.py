import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# This class is an implementation of https://github.com/keon/deep-q-learning
class CartpoolAgent:
	def __init__(self, statesize, actionsize):
		self.statesize = statesize
		self.actionsize = actionsize
		self.memory = deque(maxlen=4000)
		self.gamma = 0.85
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self._build_model()

	def _build_model(self):
		# A basic Neural network stucture
		model = Sequential()
		model.add(Dense(64, input_dim = self.statesize , activation='relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(self.actionsize, activation='linear')) # action size gives the number of actions to be taken
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
		return model

	# making a memory for replay storage
	def remember(self,state,action,reward,nextstate,done):
		self.memory.append((state, action, reward, nextstate, done))

	# this function will play an action for each state	
	def play(self , state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.actionsize) # return any action from action set
		play_score = self.model.predict(state)
		return np.argmax(play_score[0])  # returns action with largest value
	
	def replayMemory(self, batchsize=10):
		minibatch = random.sample(self.memory, batchsize)
		state_ = np.zeros((batchsize, self.statesize))
		target_ = np.zeros((batchsize , self.actionsizec))
		for state, action, reward, nextstate, done in minibatch:
			target = reward
			if not done: # suppose the game is not over and the cartpole game is continuing
				target = (reward + self.gamma * np.amax(self.model.predict(nextstate)[0]))

			playvalue = self.model.predict(state)
			playvalue[0][action] = target # here we are storing the value of target to the present q value
			state_[i,:] = state
			target_[i,:] = playvalue
		self.model.fit(state_, playvalue_, epochs=1, verbose=0)
		# reduce the value of epsilon for training	
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)	

# The reward function is actual DQN. Different from above
class CartpoolAgent_: # This class is minimizes our reward function: Q learning
	def __init__(self, statesize, actionsize):
		self.statesize = statesize
		self.actionsize = actionsize
		self.memory = deque(maxlen=4000)
		self.gamma = 0.85
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.alpha = 0.001
		self.model = self._build_model()

	def _build_model(self):
		# A basic Neural network stucture
		model = Sequential()
		model.add(Dense(32, input_dim = self.statesize , activation='relu'))
		model.add(Dense(16, activation='relu'))
		model.add(Dense(self.actionsize, activation='linear')) # action size gives the number of actions to be taken
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
		return model

	# making a memory for replay storage
	def remember(self,state,action,reward,nextstate,done):
		self.memory.append((state, action, reward, nextstate, done))

	# this function will play an action for each state	
	def play(self , state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.actionsize) # return any action from action set
		play_score = self.model.predict(state)
		return np.argmax(play_score[0])  # returns action with largest value
	
	def replayMemory(self, batchsize=10):
		minibatch = random.sample(self.memory, batchsize)
		for state, action, reward, nextstate, done in minibatch:
			target = reward
			if not done: # suppose the game is not over and the cartpole game is continuing
				target = (reward + self.gamma * np.amax(self.model.predict(nextstate)[0]))

			playvalue = self.model.predict(state)
			currvalue = playvalue[0][action]
			# Change in the reward function
			playvalue[0][action] = currvalue + self.alpha*(target - currvalue) # here we are storing the value of target to the present q value
			self.model.fit(state, playvalue, epochs=1, verbose=0)
		# reduce the value of epsilon for training	
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)