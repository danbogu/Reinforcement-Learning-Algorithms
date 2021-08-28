
from env import World
import numpy as np




env = World()



#DP solution with value iteration
states=env.get_states()
r=env.get_reward_model_new()
p=env.get_transition_model_new()
discount=0.9
th= 10**-4
V_mdp,policy_mdp= env.env_value_iteration(states,r,p,discount,th)
env.plot_value(V_mdp)
env.plot_policy(policy_mdp)




#Monte Carlo
policy_MC, Q_MC, V_MC = env.monte_carlo(30000,decay=100,alpha=0.2) 
env.plot_value(V_MC)
env.plot_policy(policy_MC)
Q_MC=np.round(Q_MC,2)
env.plot_qvalue(Q_MC)



#Sarsa
policy_SA,Q_SA,V_SA = env.sarsa(30000,decay=100,alpha=0.01)
env.plot_value(V_SA)
env.plot_policy(policy_SA)
Q_SA=np.round(Q_SA,2)
env.plot_qvalue(Q_SA)




#Q learning
policy_Q,Q_Q,V_Q = env.q_learning(30000,decay=1000,alpha=0.01)
env.plot_value(V_Q)
env.plot_policy(policy_Q)
Q_Q=np.round(Q_Q,2)
env.plot_qvalue(Q_Q)


'''
Optimization process:

def compare(q_values):
	#Given a Q_values matrix, this function calculates the sum of squared differences from the DP estimation of the same discount, 0.9
	#and returns the SSE compared to the DP estimation
    v_optimal=[0,0,-0.09,-0.124,0.408,0.054,0.085,-0.09,0.565,0, 0.178,0,0.909,0.749,0.581,0.178,0,0.909, 0.733,0] #optimal values generated in DP
    estimate = np.amax(q_values,axis=1)
    result = np.sum(np.power(np.subtract(v_optimal,estimate),2))
    return result


def plot_tuning_results(results,parameter_values,parameter):
	#This function plots the sum of squared differences on the y axis, and the parameter values on the x axis
    plt.plot(parameter_values,results)
    plt.title('Sum of Squared differences from DP solution')
    plt.xlabel(parameter)
    plt.ylabel('values')
    plt.show()


#tuning alpha parameter in SARSA
alpha_values=[0.01,0.15,0.25,0.35,0.5] #Tuning grid
results = []
for value in alpha_values:
    q, p,v = env.sarsa(num_episodes=10000,alpha=value)
    difference = compare(q)
    results.append(difference)
plot_tuning_results(results,alpha_values,'alpha')
best_parameter = alpha_values[np.argmin(results)]


#tuning epsilon parameter in SARSA based on optimal alpha
decay_values=[50,100,1000,1500] #Tuning grid
results_decay = []
for value in decay_values:
    q, p,v = env.sarsa(num_episodes=10000,decay=value,alpha=0.01)
    difference = compare(q)
    results_decay.append(difference)
plot_tuning_results(results_decay,decay_values,'decay')
best_parameter_decay = decay_values[np.argmin(results_decay)]

#tuning alpha parameter in q learning
alpha_values_q=[0.01,0.15,0.25,0.35,0.5]  #Tuning grid
results_q = []
for value in alpha_values_q:
    q, p,v = env.q_learning(num_episodes=10000,alpha=value)
    difference = compare(q)
    results_q.append(difference)
plot_tuning_results(results_q,alpha_values_q,'alpha')
best_parameter = alpha_values_q[np.argmin(results_q)]


#tuning epsilon parameter in q learning based on optimal alpha
decay_values_q=[50,100,1000,1500] #Tuning grid
results_decay_q = []
for value in decay_values_q:
    q, p,v = env.q_learning(num_episodes=10000,decay=value,alpha=0.01)
    difference = compare(q)
    results_decay_q.append(difference)
plot_tuning_results(results_decay_q,decay_values_q,'decay')
best_parameter_decay_q = decay_values_q[np.argmin(results_decay_q)]


#tuning alpha parameter in monte carlo
alpha_values_q=[0.01,0.1,0.2,0.3,0.4,0.5]  #Tuning grid
results_q = []
for value in alpha_values_q:
    q, p,v = env.monte_carlo(num_episodes=10000,alpha=value)
    difference = compare(q)
    results_q.append(difference)
plot_tuning_results(results_q,alpha_values_q,'alpha')
best_parameter = alpha_values_q[np.argmin(results_q)]


#tuning epsilon parameter in monte carlo on optimal alpha
decay_values_q=[50,100,1000,1500] #Tuning grid
results_decay_q = []
for value in decay_values_q:
    q, p,v = env.monte_carlo(num_episodes=10000,decay=value,alpha=0.2)
    difference = compare(q)
    results_decay_q.append(difference)
plot_tuning_results(results_decay_q,decay_values_q,'decay')
best_parameter_decay_q = decay_values_q[np.argmin(results_decay_q)]
'''

