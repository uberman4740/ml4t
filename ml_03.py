
# coding: utf-8

# # Machine Learning Overview
# 
# Supervised Regression Learning:
# - Numerical prediction given examples of inputs and outputs trained with data
# - As opposed to classification learning
# 
# Types of learning:
# - Linear regression (parametric)
# - K nearest neighbor (instance based)
# - Decision trees
# - Decision forests (lots and lots of decision trees taken together)
# 
# Learning with stock data:
# - given a dataframe with features of a set of stocks over time (measureable predictive factors)
# - feature data serves as the input X
# - for training, use historical prices and historical features for learning
# - use a historical price in the future from a feature and record price(t+5) vs. feature set(t) pairs as data to train the model
# - depth & breadth given by the time period over which training occurs and the stock universe we will look at
# - once model is trained, start doing predictions
# 
# Problems with regression:
# - noisy and uncertain
# - challenging to estimate confidence in forecasts
# - holding time, allocation uncertain
# - can be addressed by enforcement learning
# 
# __________

# ## Supervised Regression Learning
# - Using data to build a model that predicts a numerical outputs based on a set of numerical inputs.
# 
# #### Parametric regression: Simple Regression
# - represent the model with a number of parameters
# - for example, fitting a line to data (through linear regression on the line y = mx + b)
#     - parameters given by m and b
# - could fit, theoretically, any polynomial with additional parameters to try and describe the behavior of the data more accurately
# - in practice, much of the time you throw away the data once the model is parameterized and use that for predictions
# 
# #### Instance regression: K Nearest Neighbor
# - instead, you could use a data-centric approach where you keep the data and use it to better inform you predictions
# - find K nearest data points to the query and use them to estimate the output prediction
# - take the mean Y value of the K nearest neighbors for the prediction
# - if you repeat this process you would have a model that fits the data more appropriately
# - another similar method is kernel regression, that assigns a weight to each neighbor based on the distance from the query X value (or cartesian distance)
# - non-parametric approaches are good for models that are hard to approximate/derive mathematically, and instead are well-suited for numerical methods instead
# 
# #### Training and Testing:
# - we have data on prices and features for our stocks
# - we first want to separate testing and training data, to be able to see if the model behaves well once the model has been trained appropriately
# - take training data, put it through machine learning model to derive the parameters, then use testing data and put it through the model, and compare the output to the true prices that we know to see if the model has been successful
# - generally, train on older data and test on newer data 
# 
# #### Learning APIs:
# - will need to build api's for implementing the learners
# 
# ##### Linear Regression:
# - learner = LinRegLearner()
# - learner.train(Xtrain,Ytrain)
# - Y = learner.query(Xtest) --> compare to Ytest
# 
# <code> class LinRegLearner::
#     def __init__(self): 
#         pass
#         
#     def train(self,X,Y):
#         # fit a line to the data
#         # find an m and a b --> parameters of linear model
#         self.m, self.b = favorite_linreg(X,Y) # use algo you want from SciPy and Numpy
#         
#     def query(self,X):
#         Y = self.m*X + self.b
#         return Y
# </code>
# 
# ##### K-Nearest Neighbor:
# - learner = KNNLearner(k=3) --> arg = number of neighbors
# - learner.train(Xtrain,Ytrain)
# - Y = learner.query(Xtest) --> compare to Ytest
# 
# <code> class KNNLearner::
#     def __init__(self,k): 
#         self.k = k
#         pass
#         
#     def train(self,X,Y):
#         # find set of Y values given k for each value of X
#         # don't really have to train much
#         
#     def query(self,X):
#         Y = average Y-value of k-nearest neighbors
#         return Y
# </code>
# _____________

# ## Ensemble Learning:
# - can you combine multiple weak learners to develop one strong learner?
# - the general formulation is to generate separate models based on each machine learning algorithm, then query each model for a given input X and combine the outputs (for example through an average)
# - if doing classification, would want to use each output to assign a vote on the classification
# - why ensembles?
#     1. Lower error
#     2. Less overfitting - reduces the bias of each algorithm
# - can use boosting and bagging as wrappers on existing learners to create ensemble learners
# 
# ####  Bootstrap aggregating (bagging):
# - another method of building an ensemble learner, in which each algorithm is trained on a different subset of the data
# - Method:
#     1. split the training data into m separate subsets randomly with replacement (always choose across whole collection of the data), where the subsets {D1,...,Dm} will have n' instances from the data (original training instances is n)
#     2. Use each of the subsets to train a different model
#     3. Query each model with an equivalent X, and use the output of the models aggregated to be the ensemble prediction
# 
# #### Boosting:
# - an improvement on boostrap aggregating
# - serves to improve the learners by improving the model in places where the learning is not going well
# - after model is built, use the training data to test the model
#     - some of the input data will not be well predicted and will have an error associated with it when the model is tested
# - when we build the next subset of the data, the instance of the data will be weighted by its error factor from the first test
#     - instances with more error in the first model are more likely to get picked for the next subset than other instances
# - build another model and use the training data to test the two models, combine the two outputs and test on the training data, and use the error weight method to define the m subsets and the m models to form the overall ensemble
# - note that boosting might result in more overfit models
# 
# ________
# 

# ### Reinforcement Learning
# - Reinforcement learners create policies that provide specific direction on which action to take
# 
# #### RL Problem:
# - There are many algorithms that solve the RL problem
# - Learner perceives state S, and outputs an action a based on a policy pi(s)
# - The action a interacts with the environment, and the environment transfers to a new stat S
# - How do we arive at this policy pi(S)?
#     - the learner also has a reward function 
#     - the learner seeks to take actions that maximize the overall reward
#     - policy pi(S) is derived using an algorithm that maximises reward over time
#     
# #### Trading as an RL Problem:
# - In terms of trading, the environment is the market, and the actions are trades. The state S are factors about the stocks that we observe in the market, and the reward function is the return for making a proper trade
# - Trading as an RL Problem:
#     - buy/sell are actions
#     - holding state (long/short) are states
#     - bollinger values and statistics are states
#     - return from a trade is the reward
#     - daily return could be a state or a reward
# - Formalization of Trading as RL:
#     - States (market features, holding)
#     - Actions (BUY, SELL, DO NOTHING)
# 
# #### Markov Decision Problem:
# - Composed of:
#     - A set of states S
#     - A set of actions A
#     - A transition function T[s,a,s'] that computes the probability that given a current state s, and we take an action a, the next state will be state s'
#     - A reward function R[s,a]
# - Goal:
#     - find policy pi(s) that will maximize reward over time
# - Can use certain algorithms to determine an optimal policy, such as:
#     - policy iteration
#     - value iteration
# - However, for our problems we don't know T[.] and R[.], so we have to indirectly determine a policy given what is perceived about the environment
#     - Instead, you need to sample the environment using a selection of actions to get to new states
#     - Gathering experience tuples of states, actions, and rewards, such that you can derive some information
#     - Can then use this information for:
#         - model-based --> build a model for T[.] and R[.] and then use value iteration or policy iteration to get to the answer
#         - model-free --> such as Q-learning to develop a policy based on the data only
# - What we trying to optimize in these types of problems?
#     - First you need to think about what your horizon is with regard to the optimization
#         - infinite horizon --> sum of all rewards over all of the future
#         - finite horizon (can vary in length) --> sum of all rewards but don't go to infinity
#     -  As you change the size of your finite horizon, the optimal policy changes!
#     - Method used in Q-Learning:
#         - Can also have a discounted reward where you discount by some gamma^(i-1) so future rewards are less valuable than present rewards
#             - closer gamma is to 1.0, then the more we value the future rewards
# 
# ##### RL Summary:
# - RL algorithms solve Markov Decision Problems
# - MDPs defined by S, A, T[s,a,s'], R[s,a]
# - Goal is to find policy pi(s) that maps a state to an action we should take, that maximizes the reward over time
# 
# ________
# 
# #### Q-Learning:
# - Does not use a model, but rather builds a table of utility values as the agent interacts with the world
# - What is Q?
#     - Q function Q[s,a] is viewed as a table here where s is the state we are looking at and a is the action we might take
#     - Q is the value of taking the action a given the state s
#     - 2 components of taking action a in state s:
#         - immediate reward
#         - discounted reward (reward for future actions)
#     - as a result, Q is not greedy
# - How do you use Q table? (if you have it)
#     - given a policy pi(s), which gives the action given state s
#     - pi(s) = argmax_a{Q[s,a]} --> maximize Q given s and possible actions a
#     - After running Q-Learning long enough, you arrive at the optimal policy and optimal Q table
# 
# #### Building a Q-Table: (Training a Q-Learner)
# - General process:
#     - Select training data
#     - Iterate over time to produce experience tuples (s,a,s',r) 
#         - Use these tuples to update Q-table
#     - Test policy pi(s)
#     - Repeat until learner converges (keep going until the return converges to some high value)
# - What happens when we are iterating over the training data?
#     - Set start tim, initialize a Q table Q[.]
#     - compute the state s
#     - select the action a
#     - observe the reward and next state (r,s')
#     - update the Q-table using (s,a,s',r)
#     
# #### Update Rule:
# - How do you update the rule/policy given tuple (s,a,s',r)?
#     - Q'[s,a] = (1-alpha)Q[s,a] + alpha(improved estimate)
#     - That is, given old Q-value and improved estimate, we use a learning rate alpha (between 0 to 1.0, generally 0.2)
#     - Larger values of alpha make learning faster, and lower values make learning slower
#     - More formally:
#         - Q'[s,a] = (1-alpha)Q[s,a] + alpha(r + gamma(future r))
#         - Where alpha is the learning rate, gamma is the discount rate
#         - Higher gamma means we value future rewards MORE, while lower gamma means we value future rewards LESS
#     - How do you get the future r?
#         - future rewards = Q[s',argmax_a{Q[s',a']}
#         - given we end up in state s', if we act optimally, what would be the optimal reward along the actions available in the Q-table
# 
# - The formula for computing Q for any state-action pair (s, a), given an experience tuple (s, a, s', r), is:
# 
# <code>Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmaxa'(Q[s', a'])])</code>
# 
# - Where:
#     - r = R[s, a] is the immediate reward for taking action a in state s,
#     - γ ∈ [0, 1] (gamma) is the discount factor used to progressively reduce the value of future rewards,
#     - s' is the resulting next state,
#     - argmaxa'(Q[s', a']) is the action that maximizes the Q-value among all possible actions a' from s', and,
#     - α ∈ [0, 1] (alpha) is the learning rate used to vary the weight given to new experiences compared with past Q-values.
#     
# #### Comments:
# - Success of Q-Learning depends on exploration
# - One way to do this is using randomness, e.g. by randomly picking an action during training to see what happens such that a better Q-table can be generated
# - As training goes on, can do this with a random probability c at the beginning of learning that reduces over time, until c=0 and we use the optimal Q rule instead
# 
# ________
# 
# #### Q-Learning and Trading:
# - Actions in trading:
#     - BUY
#     - SELL
#     - DO NOTHING
# - Factors that define a State:
#     - Adjusted close/SMA (let's you compare over different stocks)
#     - PE ratio
#     - Bollinger band value
#     - Holding type (short, long, or not at all; can't sell if not holding!)
#     - return since entry (helps set exit points)
# - Given a state, the learner chooses to do an action
# - Rewards in a trading problem:
#     - could use r = daily return
#     - could use r = 0 until exit, then use r = cumulative return
# - Using r = daily return will get faster convergence for the learner, because the learning agent gets feedback on each individual action it takes, so Q-learner will update more frequently
# - Creating the state: (state is an integer, such that we can address it in the Q-table)
#     - discretize each factor
#     - combine the discretized factors
#     - e.g. four factors (X1,X2,X3,X4)
#         - discretized into four integers (0,5,9,2)
#         - stack them into discretized state 0592
# 
# #### Discretizing:
# - Need to pool data into bins that are given by a set of thresholds
# - To determine the thresholds:
# <code>
#     stepsize = size(data)/steps
#     data.sort()
#     for i in range(0,steps):
#         threshold[i] = data[(i+1)*stepsize]
# </code>
# 
# #### Recap:
# - Building a model:
#     - define states, actions, and rewards
#     - choose in-sample training period
#     - iterate and update Q-table
#     - backtest until convergence
# - Testing the model:
#     - test on new data
# - Advantages:
#     - The main advantage of a model-free approach like Q-Learning over model-based techniques is that it can easily be applied to domains where all states and/or transitions are not fully defined.
#     - As a result, we do not need additional data structures to store transitions T(s, a, s') or rewards R(s, a).
#     - Also, the Q-value for any state-action pair takes into account future rewards. Thus, it encodes both the best possible value of a state (maxa Q(s, a)) as well as the best policy in terms of the action that should be taken (argmaxa Q(s, a)).
# - Issues: 
#     - The biggest challenge is that the reward (e.g. for buying a stock) often comes in the future - representing that properly requires look-ahead and careful weighting.
#     - Another problem is that taking random actions (such as trades) just to learn a good strategy is not really feasible (you'll end up losing a lot of money!).
#     - In the next lesson, we will discuss an algorithm that tries to address this second problem by simulating the effect of actions based on historical data.
#     
# _____
# 
# #### Dyna-Q:
# - A method intended to speed up learning or model convergence for Q-learning
# - Q-learning:
#     - init Q table
#     - observe s
#     - execute a, observe s',r
#     - update Q with experience tuples (s,a,s',r)
#     - repeat until convergence
# - Dyna-Q:
#     - Learn model of T and R
#         - find new values for T and R
#         - update T'[s,a,s'] 
#         - update R'[s,a]
#     - hallucinate experience
#         - randomly select s,a
#         - infer s' from T[.]
#         - infer r from R[.]
#     - update Q-table
#         - update using (s,a,s',r) from hallucination
#     - repeat until hallucination works well
#     - Do normal Q-learning 
# 
# #### Learning T:
# - T[s,a,s'] represents the probability that in state s, taking action a will lead to state s'
# - Will use a T-table that counts the number of times one gets to state s' given state s and action a:
# <code>
#     init Tc[] = 0.00001
#     while executing, observe s,a,s'
#     increment Tc[s,a,s'] location
# </code>
# 
# - Evaluating T using T-count table:
#     - T is the probability that in state s, taking action a will lead to state s'
#     - first count Tc[s,a,s']
#     - then divide by the sum of the counts of all s,a producing states i:
#     <code>
#     T[s,a,s'] = Tc[s,a,s']/argsum_i{Tc[s,a,i]}
#     </code>
#     
# #### Learning R: 
# - R[s,a] is the expected reward for s,a, while r is the immediate reward when we experience the state transition in the real world
# - Need to update the model each time learner has a real experience:
# <code>
#     R'[s,a] = (1-α)R[s,a] + αr
#     Where α is the learning rate
# </code>
# 
# ##### Recap:
# - Overall Algorithm:
#     - Q-learning: (Q)
#         - init Q table
#         - observe s
#         - execute a, observe s',r
#         - update Q with experience tuples (s,a,s',r)
#     - Update Model
#         - T'[s,a,s'] update
#         - R'[s,a] update
#     - Hallucinate: (Dyna-Q)
#         - s = random
#         - a = random
#         - infer s' from T[.]
#         - r = R[s,a]
#     - Q-update:
#         - update Q with (s,a,s',r)
#     - Repeat Hallucinate + Q-update until convergence
#     - Repear Q-Learning until convergence
# - The Dyna architecture consists of a combination of:
#     - direct reinforcement learning from real experience tuples gathered by acting in an environment,
#     - updating an internal model of the environment, and,
#     - using the model to simulate experiences.
# 
