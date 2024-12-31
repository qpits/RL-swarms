import numpy as np
import functools
import operator
import math

class IQLCollector:
    def __init__(self, env, action_dict, actions_dict, reward_dict, qtable):
        self.environment = env
        self.action_dict = action_dict
        self.actions_dict = actions_dict
        self.reward_dict = reward_dict
        self.qtable = qtable
        self.episode = None
        self.agent = None


def avg_reward_log(data):
    logged = np.zeros(data.environment.n_chemicals)
    for idx, agent in data.environment.learners.items():
        logged[agent['type']] += data.reward_dict[str(data.episode)][str(idx)]
    logged = np.round(logged / np.array(data.environment.learner_population) / data.ticks_per_episode, 2)
    return {f'avg_rew_{i}': rew for i, rew in enumerate(logged)}


def epsilon_log(data):
    return {"epsilon": round(data.epsilon, 4)}


def avg_cluster_log(data):
    clustering = data.environment.avg_cluster2()
    return {f'avg_cluster_{i}': cluster for i, cluster in clustering.items()}

def avg_neighbourhood_entropy_log(data):
    entropies = [[] for i in range(data.environment.n_chemicals)]
    for agent_id, agent in data.environment.learners.items():
        count = np.zeros(data.environment.n_chemicals, dtype=np.int32)
        entropy = 0.
        neighbours = [data.environment.patches[p]['turtles'] for p in data.environment.cluster_patches[agent['pos']]]
        neighbours = functools.reduce(operator.iconcat, neighbours, [])
        if len(neighbours) > 0:
            for nb in neighbours:
                count[data.environment.learners[nb]['type']] += 1
            prob = count / len(neighbours)
            if np.count_nonzero(prob) > 1:
                for p in prob:
                    entropy -= p * math.log(p)
        entropies[agent['type']].append(entropy)
    entropies = np.mean(np.array(entropies), axis=1)
    return {f"avg_entropy_{i}": entropies[i] for i in range(entropies.shape[0])}

def episode_log(data):
    return {"ep": data.episode}

def actions_log(data):
    actions_ep = list(data.actions_dict[str(data.episode)].values())
    return {action: val for action, val in zip(data.environment.actions, actions_ep)}
