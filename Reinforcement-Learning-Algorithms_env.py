import numpy as np
import matplotlib.pyplot as plt
import _env
import numpy as np

class World(_env.Hidden):

    def __init__(self):

        self.nRows = 4
        self.nCols = 5
        self.stateInitial = [4]
        self.stateTerminals = [1, 2,  10, 12, 17, 20]
        self.stateObstacles = []
        self.stateHoles = [1, 2,  10, 12, 20]
        self.stateGoal = [17]
        self.nStates = 20
        self.nActions = 4

        self.observation = [4]  # initial state

    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def _plot_world(self):

        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        stateTerminals = self.stateTerminals
        stateGoal      = self.stateGoal
        coord = [[0, 0], [nCols, 0], [nCols, nRows], [0, nRows], [0, 0]]
        xs, ys = zip(*coord)
        plt.plot(xs, ys, "black")
        for i in stateObstacles:
            (I, J) = np.unravel_index(i-1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.3")
            plt.plot(xs, ys, "black")
        for i in stateTerminals:
            #print("stateTerminal", i)
            (I, J) = np.unravel_index(i-1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            #print("I,J = ", I,J)
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            #print("coord", xs,ys)
            plt.fill(xs, ys, "0.6")
            plt.plot(xs, ys, "black")
        for i in stateGoal:
            (I, J) = np.unravel_index(i-1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            #print("I,J = ", I,J)
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            #print("coord", xs,ys)
            plt.fill(xs, ys, "0.9")
            plt.plot(xs, ys, "black")
        plt.plot(xs, ys, "black")
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        plt.plot(X, Y, 'k-')
        plt.plot(X.transpose(), Y.transpose(), 'k-')



    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def plot(self):
        """
        plot function
        :return: None
        """
        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols
        self._plot_world()
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.5, j - 0.5, str(states[k]), fontsize=26, horizontalalignment='center', verticalalignment='center')
                k += 1
        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_value(self, valueFunction):

        """
        plot state value function V

        :param policy: vector of values of size nStates x 1
        :return: None
        """

        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateObstacles:
                    plt.text(i + 0.5, j - 0.5, str(self._truncate(np.round(valueFunction[k],4),3)), fontsize=16, horizontalalignment='center', verticalalignment='center')
                k += 1
        # label states by numbers
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.92, j - 0.92, str(states[k]), fontsize=11, horizontalalignment='right',verticalalignment='bottom')
                k += 1

        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_policy(self, policy):

        """
        plot (stochastic) policy

        :param policy: matrix of policy of size nStates x nActions
        :return: None
        """
        # remove values below 1e-6
        policy = policy * (np.abs(policy) > 1e-6)


        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        stateTerminals = self.stateTerminals
        #policy = policy.reshape(nRows, nCols, order="F").reshape(-1, 1)
        # generate mesh for grid world
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        # generate locations for policy vectors
        #print("X = ", X)
        X1 = X.transpose()
        X1 = X1[:-1, :-1]
        #print("X1 = ", X1)
        Y1 = Y.transpose()
        Y1 = Y1[:-1, :-1]
        #print("Y1 =", Y1)
        X2 = X1.reshape(-1, 1) + 0.5
        #print("X2 = ", X2)
        Y2 = np.flip(Y1.reshape(-1, 1)) + 0.5
        #print("Y2 = ", Y2)
        # reshape to matrix
        X2 = np.kron(np.ones((1, nActions)), X2)
        #print("X2 after kron = ", X2)
        Y2 = np.kron(np.ones((1, nActions)), Y2)
        #print("X2 = ",X2)
        #print("Y2 = ",Y2)
        # define an auxiliary matrix out of [1,2,3,4]
        mat = np.cumsum(np.ones((nStates , nActions)), axis=1).astype("int64")
        #print("mat = ", mat)
        # if policy vector (policy deterministic) turn it into a matrix (stochastic policy)
        #print("policy.shape[1] =", policy.shape[1])
        if policy.shape[1] == 1:
            policy = (np.kron(np.ones((1, nActions)), policy) == mat)
            policy = policy.astype("int64")
            print("policy inside", policy)
        # no policy entries for obstacle and terminal states
        index_no_policy = stateObstacles + stateTerminals
        index_policy = [item - 1 for item in range(1, nStates + 1) if item not in index_no_policy]
        #print("index_policy", index_policy)
        #print("index_policy[0]", index_policy[0:2])
        mask = (policy > 0) * mat
        #print("mask", mask)
        #mask = mask.reshape(nRows, nCols, nCols)
        #X3 = X2.reshape(nRows, nCols, nActions)
        #Y3 = Y2.reshape(nRows, nCols, nActions)
        #print("X3 = ", X3)
        # print arrows for policy
        # [N, E, S, W] = [up, right, down, left] = [pi, pi/2, 0, -pi/2]
        alpha = np.pi - np.pi / 2.0 * mask
        #print("alpha", alpha)
        #print("mask ", mask)
        #print("mask test ", np.where(mask[0, :] > 0)[0])
        self._plot_world()
        for i in index_policy:
            #print("ii = ", ii)
            ax = plt.gca()
            #j = int(ii / nRows)
            #i = (ii + 1 - j * nRows) % nCols - 1
            #index = np.where(mask[i, j] > 0)[0]
            index = np.where(mask[i, :] > 0)[0]
            #print("index = ", index)
            #print("X2,Y2", X2[ii, index], Y2[ii, index])
            h = ax.quiver(X2[i, index], Y2[i, index], np.cos(alpha[i, index]), np.sin(alpha[i, index]), color='b')
            #h = ax.quiver(X3[i, j, index], Y3[i, j, index], np.cos(alpha[i, j, index]), np.sin(alpha[i, j, index]),0.3)

        # label states by numbers
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.92, j - 0.92, str(states[k]), fontsize=11, horizontalalignment='right', verticalalignment='bottom')
                k += 1
        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_qvalue(self, Q):
        """
        plot Q-values

        :param Q: matrix of Q-values of size nStates x nActions
        :return: None
        """
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        stateObstacles = self.stateObstacles

        fig = plt.plot(1)

        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateHoles + stateObstacles + stateGoal:
                    #print("Q = ", Q)
                    plt.text(i + 0.5, j - 0.15, str(self._truncate(Q[k, 0], 3)), fontsize=8,
                             horizontalalignment='center', verticalalignment='top', multialignment='center')
                    plt.text(i + 0.9, j - 0.5, str(self._truncate(Q[k, 1], 3)), fontsize=8,
                             horizontalalignment='right', verticalalignment='center', multialignment='right')
                    plt.text(i + 0.5, j - 0.85, str(self._truncate(Q[k, 2], 3)), fontsize=8,
                             horizontalalignment='center', verticalalignment='bottom', multialignment='center')
                    plt.text(i + 0.1, j - 0.5, str(self._truncate(Q[k, 3], 3)), fontsize=8,
                             horizontalalignment='left', verticalalignment='center', multialignment='left')
                    # plot cross
                    plt.plot([i, i + 1], [j - 1, j], 'black', lw=0.5)
                    plt.plot([i + 1, i], [j - 1, j], 'black', lw=0.5)
                k += 1

        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def get_nrows(self):

        return self.nRows

    def get_ncols(self):

        return self.nCols

    def get_stateTerminals(self):

        return self.stateTerminals

    def get_stateHoles(self):

        return self.stateHoles

    def get_stateObstacles(self):

        return self.stateObstacles

    def get_stateGoal(self):

        return self.stateGoal


    def get_nstates(self):

        return self.nStates

    def get_nactions(self):

        return self.nActions


    def step(self,action):

        nStates = self.nStates
        stateGoal = self.get_stateGoal()
        stateTerminals = self.get_stateTerminals()

        state = self.observation[0]


        # generate reward and transition model
        p_success = 0.8
        r = -0.04
        self.get_transition_model(p_success)
        self.get_reward_model(r,p_success)
        Pr = self.transition_model
        R = self.reward
        prob = np.array(Pr[state-1, :, action])
        #print("prob =", prob)
        next_state = np.random.choice(np.arange(1, nStates + 1), p = prob)
        #print("state = ", state)
        #print("next_state inside = ", next_state)
        #print("action = ", action)
        reward = R[state-1, next_state-1, action]
        #print("reward = ", R[:, :, 0])
        observation = next_state

        #if (next_state in stateTerminals) or (self.nsteps >= self.max_episode_steps):
        if (next_state in stateTerminals):
            done = True
        else:
            done = False

        self.observation = [next_state]


        return observation, reward, done


    def reset(self, *args):
        
        nStates = self.nStates
        if not args:
            observation = self.stateInitial
        else:
            observation = []
            while not (observation):
                observation = np.setdiff1d(np.random.choice(np.arange(1, nStates +  1, dtype = int)), self.stateHoles + self.stateObstacles + self.stateGoal)
        self.observation = observation



    def render(self):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        observation = self.observation #observation
        state = observation[0]


        J, I = np.unravel_index(state - 1, (nRows, nCols), order='F')



        J = (nRows -1) - J



        circle = plt.Circle((I+0.5,J+0.5), 0.28, color='black')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle)

        self.plot()


    def close(self):
        plt.pause(0.3) #0.5
        plt.close()

    def show(self):
        plt.ion()
        plt.show()

        
        
        ######Our_Code:
    
    #Transition Model
    def get_transition_model_new(self):
    	'''
    	This method returns env transition model by the following rules:
    	-Four possible actions (0-3).
    	-The env is stochastic with 80% chance to move to desired direction
    	and 10% chance to slide sideways. (represented by the 'actions_move' dict).
		returns: transition model.
    	'''
        actions_move = {0 : {(0,-1) : 0.8, (1,0) : 0.1, (-1,0) : 0.1},
                        1 : {(1,0) : 0.8, (0,-1) : 0.1, (0,1) : 0.1},
                        2 : {(0,1) : 0.8, (1,0) : 0.1, (-1,0) : 0.1},
                        3 : {(-1,0) : 0.8, (0,-1) : 0.1, (0,1) : 0.1},}
        transition_model = np.zeros((self.nActions,self.nStates,self.nStates)) 
        valid_states = [x for x in range(1,self.nStates+1) if x not in self.stateGoal and x not in self.stateHoles]

        for action in range(self.nActions):
            for row in range(self.get_nrows()):
                for col in range(self.get_ncols()):
                    position = row + col * self.get_nrows() #transform grid (row x col) position to sequential state number.
                    if position + 1 in valid_states:
                        for move in actions_move[action]:
                        	#check to verify next actions is valid(within the states space)
                            if 0 <= (row + move[1]) < self.get_nrows() and 0 <= (col + move[0]) < self.get_ncols():
                                _to_state = (row + move[1]) + (col + move[0]) * self.get_nrows() #next state after movement
                                #update transition probability from state to next state for spesific action
                                transition_model[action][position][_to_state] += actions_move[action][move] 
                            else:
                            	#update transition probability from state to same state (bounce back - due to invalid action for states space)
                                transition_model[action][position][position] += actions_move[action][move]
                    else:
                        transition_model[action][position][position] =1 

        #Markov transition matrix w/ all rows sum to 1
        for n in range(self.nActions):
             assert np.sum(np.sum(transition_model[n],axis=1))==self.nStates,f"Matrix of action {n} doesn't sum to 20, not all the rows sum to 1" 

        return transition_model
    
    #Reward Model
    def get_reward_model_new(self):
    	'''
    	This method returns a reward movel for env.
    	'''
        # Create reward vector
        R = np.zeros((self.nStates,1)) -0.04 
        #add -1 to each hole state
        for i in self.stateHoles:
            R[i-1] += -1
        #each 1 to the goal state
        R[self.stateGoal[0]-1] += 1
        return R

    #get states function
    def get_states(self):
        states = range(self.nStates)
        return states
    
    #Clause 1- Value iteration method for the DP solution
    def env_value_iteration(self,states,reward,transition,discount,th):
    	'''
    	This method performes an Asynchronous dynamic programming, value iteration algorithm.
    	params: env states, reward matrix, transition matrix, discount factor and thetha factor.
    	returns: value function, policiy.
    	'''
        #set Values vector, flag for final values and counter iterations
        V = np.zeros((self.nStates,1))
        final_values=False
        count_iteration=0
        while not final_values:
            delta=0
            for s in states:
                #save the old values
                v = np.copy(V[s])
                #set the vector of the possible next actions for each state 
                possible_next=np.zeros((self.nActions,1))
                for a in range(self.nActions):
                    #set the value of a specific action in a specific state based on the transition, reward and the current value-V
                    possible_next[a]=np.dot(transition[a][s],reward)+(discount*np.dot(transition[a][s],V))
                    possible_next[a]=np.round(possible_next[a],3)
                if s not in [0,1,9,11,16,19]:
                    # if s not a hole, set the value of this state to be the maximum from all it's possible actions
                    V[s] = np.amax(possible_next)
                #set the delta to be the maximum untill now in this iteration
                delta=np.maximum(delta,abs(v-V[s]))
                count_iteration+=1
            #if the delta is less then the TH, we get the final values 
            if delta<th:
                final_values=True
        #policy vector
        policy = np.zeros((self.nStates,1))
        #options vector
        options = np.zeros((self.nStates,4))
        for i in range(self.nActions):
            #set all the options to this action in all states based on the transitions nd the final values-V
            options[:,[i]] = np.dot(transition[i],reward) + discount * np.dot(transition[i],V)
        #set the policy to be the maximum option 
        policy = np.argmax(options,axis=1)
        policy=policy.reshape(-1, 1)
        #add one as the plot policy function gets 1-4 
        policy = policy + 1
        print(f"Value Iteration found optimal values and policy for discount {discount} and threshold {th} in {count_iteration} iterations")
        return V,policy
    
    def initiate_action_value(self,nStates, nActions):
    	'''
    	This method initiates a zero valued action-value function given number of actions and states in env.
    	'''
        return np.zeros((nStates,nActions))

    def initiate_uniform_policy(self,nStates, nActions):
    	'''
    	This method initiates uniform probability policy given number of actions and states in env.
    	'''
        return np.full((nStates,nActions),1/nActions)

    def next_action(self,policy, current_state, greedy = False):
    	'''
    	This method returns next action to take given a policy by choosing an action based on policy probability.
    	params: policy, current state of agent, **optional greedy=True returns the actions most preferred by policy. 
    	'''
        n_actions = policy.shape[1]
        if greedy:
            return np.argmax(policy[current_state])
        return np.random.choice(list(range(n_actions)),p = policy[current_state])

    def update_action_value(self,Q,state,action,a_s_return,alpha):
    	'''
    	This method updates the action-value function of a given state and action,
    	and alpha parameter that being multiplied with the differece of current value and a given state return.
    	returns: updated state-action function.
    	'''
        current_value = Q[state-1,action]
        #Use constat alpha learning rate
        Q[state-1,action] = current_value + alpha*(a_s_return - current_value)
        return Q

    def update_policy_epsilon_greedy(self,policy, Q, epsilon):
    	'''
    	This method updates the policy by epsilon-greedy method given an epsilon and a state-value function.
    	The policy is updated by giving 1-epsilon probability to highest value actions and epsilon for other.
    	returns: updated policy.
    	'''
        for s in range(len(Q)):
            greedy_action = np.argmax(Q[s])
            fraction = epsilon / len(Q[s])
            for a in range(len(Q[s])):
                if a == greedy_action:
                    policy[s][a] = 1 - epsilon + fraction
                    continue
                else:
                    policy[s][a] = fraction
        return policy

    def get_value_function_state_value(self,Q):
    	'''
    	This method returns a vector in the size of states space 
    	with value function (by best action) for each state given a state-value function.
    	'''
        V = []
        for s in range(len(Q)):
            best_value = max(Q[s])
            V.append(best_value)
        return V

    def transform_policy_to_greedy(self,policy):
    	'''
    	This method transformes a given policy to a greedy policy,
    	by returning a vector of actions that are most preferred by policy.
    	'''
        p = np.zeros((self.nStates,1))
        for s in range(len(policy)):
            p[s]=np.argmax(policy[s])+1
        p=p.reshape(-1, 1)   
        return p
    

#monte-carlo:
    
    def monte_carlo(self, num_episodes,decay = 100,alpha = 0.2, gamma = 0.9):
    	'''
    	This method performes a Monte-Carlo GLIE, first-visit, alpha-constant control algorithm.
    	params: number of episodes, decay-rate,alpha and gamma are set as default to best chosen parameters after tuning.
    	returns: greedy policy, state - value function, value function.
    	'''
    	#set the Q matrix
        Q = self.initiate_action_value(self.nStates,self.nActions)
        #set the initial policy to be uniform policy
        policy = self.initiate_uniform_policy(self.nStates,self.nActions)
        for i in range(num_episodes):
            first_visit_return = {}
            self.reset()
            state =self.observation[0] #Get current state
            episode = []
            episode_rewards = []
             #define epsilon based on the decay
            epsilon = 1/(i/decay+1) 
            #set policy to greedy policy
            policy = self.update_policy_epsilon_greedy(policy, Q, epsilon)
            done = False
            while(not done): #We don't count final step in the episode
            	#choose next acion
                action = self.next_action(policy,state - 1) 
                #make a state-action tuple
                q = (state, action) 
                #make a step
                state, reward, done = self.step(action)
                episode.append(q)
                episode_rewards.append(reward)
            for s_a in range(len(episode)):
            	#check for first visit (logic - check for occurrence of state-value previously in the episode - if exists now is not first visit).
                if (episode[s_a] not in episode[:s_a]): 
                    #First reward is for move 1->2 so we want reward for move from state s till end
                    state_return = sum([episode_rewards[i+s_a] * gamma ** i for i in range(len(episode_rewards[s_a:]))])#calculate return for first visit state-action
                    first_visit_return.update({episode[s_a]:round(state_return,3)})
            for s_a in first_visit_return:
            	#update action-value function for all first visited states
                Q = self.update_action_value(
                    Q = Q, state = s_a[0], action = s_a[1],a_s_return = first_visit_return[s_a],alpha = alpha)
        V = self.get_value_function_state_value(Q)
        greedy_policy = self.transform_policy_to_greedy(policy)
        print(f"found the best policy for { num_episodes} episodes" )
        return greedy_policy, Q, V
    
#Sarsa    
    
    def sarsa(self,num_episodes,decay=100,alpha = 0.01, gamma = 0.9):
    	'''
    	This method performs a SARSA algorithm.
    	params:number of episodes, decay-rate,alpha and gamma are set as default to best chosen parameters after tuning.
    	returns: greedy policy, state - value function, value function.
    	'''
        #set the Q matrix
        Q = self.initiate_action_value(self.nStates,self.nActions)
        #set the initial policy to be uniform policy
        policy = self.initiate_uniform_policy(self.nStates,self.nActions)
        for i in range(num_episodes):
            #define epsilon based on the decay
            epsilon = 1/(i/decay+1) 
            self.reset()
            t = 0
            states = self.observation
            done = False
            while (not done):
                #set policy to greedy policy
                policy = self.update_policy_epsilon_greedy(policy, Q, epsilon)
                #generate next action 
                action = self.next_action(policy,states[t] - 1)
                #make a step
                next_state, reward, done = self.step(action)
                #save the new state
                states = np.append(states,next_state)
                if(not done):
                    #generate one more action for t+1
                    action_next = self.next_action(policy,states[t+1]-1)
                    #update Q function
                    Q[states[t]-1,action]= Q[states[t]-1,action]+alpha*(reward+gamma*Q[states[t+1]-1,action_next]-Q[states[t]-1,action])
                    t += 1
                else:
                    #All missing terms are set to zero and n-step (1-step) return is equal to ordinary return.
                    Q[states[t]-1,action]= Q[states[t]-1,action]+alpha*(reward+gamma*0-Q[states[t]-1,action])
        #get value function
        V = self.get_value_function_state_value(Q)
        #get policy
        greedy_policy = self.transform_policy_to_greedy(policy)
        print(f"found the best policy for { num_episodes} episodes" )
        return Q,greedy_policy,V
    
#Q_Learning
    def q_learning(self,num_episodes,decay=1000,alpha = 0.01, gamma = 0.9):
    	'''
    	This method performs a Q Learning algorithm.
    	params:number of episodes, decay-rate,alpha and gamma are set as default to best chosen parameters after tuning.
    	returns: greedy policy, state - value function, value function.
    	'''
        #set the Q natrix
        Q = self.initiate_action_value(self.nStates,self.nActions)
        #set the initial policy to be uniform policy
        policy = self.initiate_uniform_policy(self.nStates,self.nActions)
        for i in range(num_episodes):
            #define epsilon based on the decay
            epsilon = 1/(i/decay+1) 
            self.reset()
            t = 0
            states = self.observation
            done = False
            while (not done):
                #set policy to greedy policy
                policy = self.update_policy_epsilon_greedy(policy, Q, epsilon)
                #generate next action 
                action = self.next_action(policy,states[t] - 1)
                #make a step
                next_state, reward, done = self.step(action)
                 #save the new state
                states = np.append(states,next_state)
                if(not done):
                    #choose the action as the max action
                    action_next = self.next_action(policy,states[t+1]-1,greedy = True)
                    #update Q of state in time t according to this max action
                    Q[states[t]-1,action] = Q[states[t]-1, action] + alpha * (reward + gamma * Q[states[t+1]-1, action_next] -Q[states[t]-1,action])
                    t += 1
                else: #All missing terms are set to zero and n-step (1-step) return is equal to ordinary return.
                    Q[states[t]-1, action] = Q[states[t]-1, action] + alpha * (reward + gamma * 0 - Q[states[t]-1, action])
         #get value function
        V = self.get_value_function_state_value(Q)
        #get policy
        greedy_policy = self.transform_policy_to_greedy(policy)
        print(f"found the best policy for { num_episodes} episodes" )
        return Q,greedy_policy,V

