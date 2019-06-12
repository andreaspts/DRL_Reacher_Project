# Project report

## Aspects of the learning algorithm

### Description of the DDPG (deep deterministic policy gradient) algorithm

The employed learning algorithm is the DDPG algorithm which was introduced in the articles [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf) and [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) to solve [Markov Decision Processes](https://en.wikipedia.org/wiki/Markov_decision_process) with continuous action spaces.

At the heart of the learning agent are two deep neural networks which act as concurrent function approximators, i.e., they learn a Q-function (in an off-policy way via the Bellman equation) and a policy (via the Q-function) in parallel. Hence, it is a so-called [actor-critic approach](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f) where the actor is represented by the first and the critic by the second network.


1. Fed with a state `s`, the actor network returns an action `a`. More precisely, the actor network approximates the optimal deterministic policy which implies that the returned action is the best one for any fed-in state.

2. In the next step, we feed the state `s` and the action `a` into the critic network which returns the Q-value. In this way, the critic approximates the optimal action-value function through the input of the actor’s best action.

3. Both networks have target networks (as in the standard deep Q-learning algorithm) to stabilize the algorithm. The respective target networks are copied over from their main counterpart networks every update step. (This could also be changed to an update every `4` steps.)

4. 

which is be taken and fed back into the algorithm as a reinforcement signal.

The DQL algorithm has two major processes which are closely intertwined. In the first, we sample the environment by performing actions and store the observed experience tuples in a replay memory. In particular, within each episode and for every subordinate timestep,
- we choose an action a from a state s using an epsilon-greedy policy (and the latter is obtained via the Q-table)
- we take an action a, observe the reward r
- we prepare the next state s'
- we store the experience tuple <s,a,r,s'> in the replay memory 
- we set s' to s

In the second process, we randomly select a small batch of tuples from this memory and learn from that batch via a gradient descent update step. (For this we actually need a local network with weights **w** and a target network with weights **w-**. The target network is a separate network the weights of which (**w-**) are not changed during the learning step.) In particular,
- we set a target (using the target action-value weights **w-**)
- with this we perform the gradient descent in the local network 
- at a fixed number of steps we reset the weights and obtain new **w-**'s.

More specifically, in the replay buffer we store experience tuples <s, a, r, s'> up to a particular buffer size. The tuples are added gradually step by step, episode by episode to the buffer. For the gradient descent update we use MSELoss loss function (aka L2 loss function) and an Adam optimizer. The latter is also fed with the learning rate determining the speed of the gradient descent.  A "soft update" of the model parameters connects the local and target model and is responsible for resetting of weights of the target network.


These points are clarified by following the pseudo-algortihm as taken from the [original literature](https://arxiv.org/abs/1509.02971) is depicted below

<p align="center">
  <img width="460" height="300" src="algorithm.png">
</p>

### Network architecture
Due to the fact that we are using state vectors as an input and not image data we use a simple deep neural network instead of a convolutional neural network to determine the action-value function. The former consists of the following 5 layers coded into the model.py file:

- Fully connected layer - input: 37 (state size) output: 64
- Fully connected layer - input: 64 output: 32
- Fully connected layer - input: 32 output: 16
- Fully connected layer - input: 16 output: 8
- Fully connected layer - input: 8 output: 4 (action size)

### Specification of parameters used in the Deep Q-Learning algorithm
We speficy the parameters used in the Deep Q-Learning algorithm (as in the dqn-function of the Navigation_solution.ipynb notebook):

- We set the number of episodes n_episodes to `2000`. The number of episodes needed to solve the environment and reach a score of `30.0` is expected to be smaller.
- We set the maximum number of steps per episode max_t to `1500`.

Furthermore, we give the parameters used in the ddpg_agent.py file:

- The size of the replay buffer BUFFER_SIZE is set to `10^6`.
- The mini batch size BATCH_SIZE is set to `128`.
- The discount factor GAMMA for future rewards is set to `0.99`.
- We set the value for the soft update of the target parameters TAU to `10^-3`.
- The learning rate for the gradient descent for the actor network LR_ACTOR is set to `2 * 10^-4`.
- The learning rate for the gradient descent for the critic network LR_CRITIC is set to `2 * 10^-4`.
- The parameter to control the L2 weight decay WEIGHT_DECAY is set to `0`.


## Results

With the above specifications we create a training run and report the results.

First we give a plot of the scores over the episodes:

<p align="center">
  <img width="460" height="300" src="plot1.png">
</p>

Therein, we applied a simple and exponential moving average function at window sizes of `3` (green plot and yellow plot, respectively) overlaying the original data (blue). The red line indicates the threshold of `30.0` reward points. More information on how to construct these moving averages in python can be found under the following links:
[Moving average in python](https://www.quora.com/How-do-I-perform-moving-average-in-Python) and [Exponential Moving average in python](https://www.youtube.com/watch?v=3y9GESSZmS0). Notice that the exponential moving average gives more emphasis to recent data than the simple version of it. 
In general, [moving averages](https://en.wikipedia.org/wiki/Moving_average) are a method to smoothen time and data series. 
Yet another plot (in magenta) depicts a running average with a Gaussian type of window which provides a much better smoothening as compared to the others (at least up to closely before the last episodes).
The steep fall off is due to the parameters chosen in the Gaussian averaging function and shall not be of any concern here.

Then we list the average score every `100` episodes up to the point where the agent reaches a score equal or higher than `30.0`: 

```
Episode 185	Average Score: 28.70	Score: 31.16
Episode 186	Average Score: 28.83	Score: 22.17
Episode 187	Average Score: 28.95	Score: 21.46
Episode 188	Average Score: 29.10	Score: 28.78
Episode 189	Average Score: 29.24	Score: 32.95
Episode 190	Average Score: 29.41	Score: 38.00
Episode 191	Average Score: 29.64	Score: 31.10
Episode 192	Average Score: 29.63	Score: 25.21
Episode 193	Average Score: 29.77	Score: 35.09
Episode 194	Average Score: 29.95	Score: 28.50
Episode 195	Average Score: 29.90	Score: 22.82
Episode 196	Average Score: 29.83	Score: 24.78
Episode 197	Average Score: 30.06	Score: 32.05

Environment solved in 197 episodes!	Average Score: 30.06
```

## Possible extensions of the setting and future work

1. The hyperparameters should be optimized: For example, we could change the learning rate, the batch size and improve the network structure (more/less layers and units; overfitting could be tackled using dropout or L2 regularization).

2. While the original DDPG paper advocated the use of time-correlated [Ornstein-Uhlenbeck noise](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process), it has been suggested to apply uncorrelated, mean-zero Gaussian noise. The learning behavior with these different noises could be compared.

3. The training could be repeated for the version `2` of the environment to check how the 'learning behavior' is changed 

4. [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) could be implemented. In this work the authors extend the idea of experience replay. They introduce a method which prioritizes experiences by replaying important transitions more often which accelerates the learning rate. 

5. The environment could be solved using different algorithms. In particular, 

5.a. more stable methods are the Trust Region Policy Optimization (TRPO) and Truncated Natural Policy Gradient (TNPG) which were introduced [here](https://arxiv.org/abs/1604.06778).

5.b the second version of the environment could be tackled with the [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) algorithms. They can distribute the work of gathering experience to multiple (non-interacting) copies of the same agent.
