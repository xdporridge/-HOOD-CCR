import bisect
import math

import numpy as np
import torch

class MiddleMan:
    def __init__(self, trigger, target, dist, p_steps = 10, p_rate = .01):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.p_rate = p_rate
        self.p_steps = p_steps
        self.steps = 1

        self.min = 1
        self.max = -1
        
    def __call__(self, state, action, reward, prev_act):
        with torch.no_grad():
            if self.steps > 1 or torch.rand(1) <= self.p_rate :
                poisoned = self.trigger(state)
                reward_p = self.dist(self.target, action)
                self.steps = self.steps + 1 if self.steps < self.p_steps else 1
                return poisoned, reward_p, True
            return state, reward, False
        
class BadRLMiddleMan:
    def __init__(self, trigger, target, dist, p_rate, Q, source = 2, strong = False):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.p_rate = p_rate
        self.steps = 0
        self.p_steps = 0
        self.Q = Q
        self.strong = strong
        self.source = source

        self.queue = []

    def time_to_poison(self, obs):
        with torch.no_grad():
            self.steps += len(obs)
            if self.p_steps / self.steps < self.p_rate:
                scores = self.Q(obs).cpu()
                for i in range(len(obs)):
                    score = torch.max(scores[i]).item() - scores[i][self.target]
                    bisect.insort(self.queue, score)
                    percentile_index = int(len(self.queue) * (1 - self.p_rate))
                    if score > self.queue[percentile_index]:
                        self.p_steps += 1
                        if self.strong:
                            if self.steps % 2 == 0:
                                action = np.random.choice(
                                    np.array([
                                        j for j in range(len(scores[i]))
                                        if j != self.target
                                    ])
                                )
                            else:
                                action = self.target
                        else:
                            action = None
                        return True, i, action
            return False, -1, None
    
    def obs_poison(self, state):
        with torch.no_grad():
            return self.trigger(state)
    
    def reward_poison(self, action):
        with torch.no_grad():
            return self.dist(self.target, action)
        
class DeterministicMiddleMan:
    def __init__(self, trigger, target, dist, total, budget):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.budget = budget
        self.index = int(total/budget)
        self.steps = 0

    def time_to_poison(self, obs):
        n = len(obs)
        old = self.steps
        self.steps += n
        if (old//self.index) != (self.steps//self.index):
            return True, n - (self.steps%self.index) - 1, None
        return False, -1, None
    
    def obs_poison(self, state):
        with torch.no_grad():
            return self.trigger(state)
    
    def reward_poison(self, action):
        with torch.no_grad():
            return self.dist(self.target, action)


class UntargetedMiddleMan:
    """Periodic TrojDRL poisoning for an untargeted policy backdoor.

    A strong attacker replaces poisoned actions with samples from U(A), while a
    weak attacker preserves the policy action. Both assign the same positive
    reward to every poisoned state-action pair.
    """

    def __init__(self, trigger, reward, total, budget, n_actions, strong=False):
        if total <= 0:
            raise ValueError("total must be positive")
        if budget <= 0:
            raise ValueError("budget must be positive")
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")

        self.trigger = trigger
        self.reward = float(reward)
        self.n_actions = int(n_actions)
        self.strong = strong

        self.total = int(total)
        self.budget = min(self.total, max(1, int(round(budget))))
        self.steps = 0
        self.poisoned = 0

    def _pending_indices(self, n):
        old_steps = self.steps
        new_steps = old_steps + n
        poisoned = self.poisoned
        indices = []
        while poisoned < self.budget:
            poison_step = math.ceil(
                (poisoned + 1) * self.total / self.budget
            ) - 1
            if poison_step >= new_steps:
                break
            poisoned += 1
            if poison_step >= old_steps:
                indices.append(poison_step - old_steps)
        return indices

    def peek_poisoning(self, obs):
        """Return due batch indices without advancing the schedule or RNG."""
        return self._pending_indices(len(obs))

    def plan_poisoning(self, obs):
        """Return every (batch index, replacement action) due in this vector step."""
        n = len(obs)
        poison_indices = self._pending_indices(n)
        self.steps += n
        self.poisoned += len(poison_indices)
        plan = []
        for poison_index in poison_indices:
            poison_action = None
            if self.strong:
                poison_action = int(np.random.randint(self.n_actions))
            plan.append((poison_index, poison_action))
        return plan

    def time_to_poison(self, obs):
        """Compatibility helper when at most one event can fall in a vector step."""
        if len(self.peek_poisoning(obs)) > 1:
            raise RuntimeError(
                "multiple poison events are due; use plan_poisoning for vector batches"
            )
        plan = self.plan_poisoning(obs)
        if not plan:
            return False, -1, None
        poison_index, poison_action = plan[0]
        return True, poison_index, poison_action

    def obs_poison(self, state):
        with torch.no_grad():
            return self.trigger(state)

    def reward_poison(self, action):
        del action
        return self.reward


class MiddleMan_SleeperNets:
    def __init__(self, trigger, target, dist, total, budget):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.budget = budget
        self.index = int(total/budget)
        self.steps = 0

    def time_to_poison(self):
        self.steps += 1
        return self.steps%self.index == 0
    
    def obs_poison(self, state):
        with torch.no_grad():
            return self.trigger(state)
    
    def reward_poison(self, action, V):
        with torch.no_grad():
            return self.dist(self.target, action)
	
def SimpleSelection(length, p_rate, poisoned, observed):
    probs = np.ones(length)/length
    indices = np.random.choice(np.arange(0, length, 1), int(np.ceil(length*p_rate)), replace = False, p = probs)
    indices = torch.tensor(indices).long()
    temp = list(indices)
    temp.sort()
    return torch.tensor(temp)

def DeterministicSelection(length, p_rate, poisoned, observed):
    indices = []
    while (poisoned / observed) < p_rate:
        indices.append(np.random.randint(0, length))
        poisoned += 1
    indices.sort()
    return torch.tensor(indices)


class BufferMan_Simple:
    def __init__(self, trigger, target, dist, alpha = 0.5, p_rate = .01, simple = True):
        self.trigger = trigger
        self.target = target
        self.dist = dist
        self.p_rate = p_rate
        self.alpha = alpha
        self.poisoned = 0
        self.observed = 0
        if simple:
            self.select = SimpleSelection
        else:
            self.select = DeterministicSelection
    def __call__(self, states, actions, rewards, values, logs, gamma, agent):
        self.observed += len(states)
        indices = self.select(len(states), self.p_rate, self.poisoned, self.observed)
        self.poisoned += len(indices)
        avg_perturb = 0

        if len(indices) > 0:
            states[indices] = self.trigger(states[indices])
            _, adv_log, _, adv_value = agent.get_action_and_value(states[indices], actions[indices])
            values[indices] = adv_value[:,0]
            logs[indices] = adv_log

            rtg = 0
            indice = -1
            for index in reversed(range(len(rewards))):
                rtg = rewards[index] + (gamma * rtg)
                if index == indices[indice]:
                    old_reward = rewards[index].item()
                    rewards[index] = self.dist(self.target, actions[index:index+1]) - (self.alpha * (rtg - old_reward))
                    avg_perturb += torch.absolute(rewards[index] - old_reward)
                    if (indice*-1) < len(indices) and index-1 == indices[indice-1]:
                        indice -= 1
                elif index == indices[indice] - 1:
                    if (indice*-1) < len(indices):
                        indice -= 1
                    rewards[index] = rewards[index] - (gamma * rewards[index + 1]) + (gamma * old_reward)
                    avg_perturb += torch.absolute(-(gamma * rewards[index + 1]) + (gamma * old_reward))
        return states, rewards, indices, avg_perturb

class SingleValuePoison:
    def __init__(self, indices, value):
        self.indices = indices
        self.value = value

    def __call__(self, state):
        index = self.indices
        poisoned = torch.clone(state)
        if len(state.shape) > 1:
            poisoned[:, index] = self.value
        else:
            poisoned[index] = self.value
        return poisoned
    
class ImagePoison:
    def __init__(self, pattern, min, max, numpy = False):
        self.pattern = pattern
        self.min = min
        self.max = max
        self.numpy = numpy

    def __call__(self, state):
        if self.numpy:
            poisoned = np.float64(state)
            poisoned += self.pattern
            poisoned = np.clip(poisoned, self.min, self.max)
        else:
            poisoned = torch.clone(state)
            poisoned += self.pattern
            poisoned = torch.clamp(poisoned, self.min, self.max)
        return poisoned

class Discrete:
    def __init__(self, minimum=-1, maximum=1):
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def __call__(self, target, action):
        action_array = (
            action.detach().cpu().numpy()
            if isinstance(action, torch.Tensor)
            else np.asarray(action)
        )
        if action_array.size != 1:
            raise ValueError("Discrete reward expects exactly one action.")
        action_value = action_array.reshape(-1)[0].item()
        return self.maximum if target == action_value else self.minimum
