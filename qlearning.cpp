#include <mlpack/core.hpp>
#include <iostream>
using namespace mlpack ; // include mlpack package
// Includes necessary headers
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp> // for gaussian initializa1tion
#include <mlpack/methods/reinforcement_learning/q_learning.hpp> // use Q learning
#include <mlpack/methods/reinforcement_learning/replay/random_replay.hpp> // use Replay memory
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp> // Environment Cart Pole
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp> // Use greedy Policy
#include <mlpack/core/optimizers/adam/adam_update.hpp> // Adam Optimers used
#include <mlpack/methods/reinforcement_learning/training_config.hpp> // To supply trainig config params
#include <mlpack/methods/ann/ffn.hpp> // for neural network 
#include <mlpack/methods/ann/layer/layer.hpp> // adding layer 

using namespace mlpack::ann; // namespace for artificial neural network
using namespace mlpack::optimization; //using adam optimizer . Various others are difined in mlpack
using namespace mlpack::rl; // for reinforcement leanring


using namespace std; 
int main(int argc, char const *argv[])
{
	
	// FFN stands for feed forward network . Our network takes 4 input from the observation space. 
	// This takes Cart position, Cart Velocity , Pole Angle and Pole's tip Velocity. 
	// You may refer to https://github.com/openai/gym/wiki/CartPole-v0 for more details.
	// Our network consists of 2 hidden layer and a final layer. 
	// Hidden layers have 64 neurons and 32 neurons respectively. 
	// Our obejective is to get the Q value of taking left and right respectively.
	FFN<MeanSquaredError<>,GaussianInitialization> mymodel(MeanSquaredError<>() ,GaussianInitialization(0, 0.001));
	mymodel.Add<Linear<>>(4, 64); // 4 is the observation space . 
	mymodel.Add<ReLULayer<>>();
	mymodel.Add<Linear<>>(64, 32); 
	mymodel.Add<ReLULayer<>>();
	mymodel.Add<Linear<>>(32, 2);

	// Network has been defined above
	// Now we will define policy and environment
	// We are using Cartpole Environment
	// CartPole class has been defined in cart_pole.hpp

	// Q learning plays greedy on Q values. Hence our policy is greedy. You can change policy or 
	// define your policy. You may want to refer to https://github.com/mlpack/mlpack/tree/master/src/mlpack/methods/reinforcement_learning/policy
	GreedyPolicy<CartPole> policy(1.0, 1000, 0.1);

	// Now we will define memory for our agent . We need to tell our agent batch_size and memorysize .
	// You will note that both greedypolicy and randomreply takes environment type as a template parameter. 
	int batch_size = 20;
	int replaySize = 10000;
	RandomReplay<CartPole> replayMethod(batch_size, replaySize);

	// we now need to set the training config params
	TrainingConfig config;
	config.StepSize() = 0.01;
	config.Discount() = 0.99;
	config.TargetNetworkSyncInterval() = 100;
	config.ExplorationSteps() = 100;
	config.DoubleQLearning() = false;
	config.StepLimit() = 400;

	// decltype is used to pass the type of object defined
	// std::move is used for transfer of object
	// refer : http://en.cppreference.com/w/cpp/utility/move
	// Our agent has been defined . It will learn the game via DQN
	QLearning<CartPole , decltype(mymodel) , AdamUpdate , decltype(policy)>  agent(std::move(config), std::move(mymodel), std::move(policy),
          std::move(replayMethod));
	// Our agent has now been intitialized above. You should note how easy it now becomes for the agent to tell environment , model ,  optimizer ,policy
	// Using template metaprogramming essentially helps us


	arma::running_stat<double> averageReturn;
    size_t episode = 0;
    size_t maxiter = 1000;
    size_t requirement = 50; // This variable checks if the game is converging or not .
	// References for armadillo running_stat : http://arma.sourceforge.net/docs.html#running_stat
	int i = 0;
	while ( episode <= maxiter)
	{
		double epi_return = agent.Episode();
		averageReturn(epi_return);
		episode = episode + 1;
	    std::cout << "Average return: " << averageReturn.mean()<< " Episode return: " << epi_return<< std::endl;
	    if (averageReturn.mean() > requirement)
	    {
	    	agent.Deterministic() = true;
	    	arma::running_stat<double> testReturn; // check the stats for test run to take place
	    	
	    	for (size_t i = 0; i < 20; ++i)// 20 test runs
		        testReturn(agent.Episode()); // variable defined above

		    std::cout << endl <<"Converged with return " <<  testReturn.mean()  << " with number of " << episode << " iterations"<<endl;
		    break;
	    }	
	    // check converged or not?
	}
	if (episode >= maxiter)
	{
		std::cout << "Cart Pole with DQN failed to converge in " << maxiter << " iterations." <<std::endl;
	}
	return 0;
}
