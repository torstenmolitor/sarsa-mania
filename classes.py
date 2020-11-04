import numpy as np
from itertools import count
import copy
from numpy.random import choice


class StateSpace:
    def __init__(self, grid_size, r, p, b, r_epsilon, p_epsilon):
        self.grid_size = grid_size
        self.robber_initial_position = r
        self.police_initial_position = p
        self.bank_position = b

        self.initial_state = None

        self.robber_epsilon = r_epsilon
        self.robber_states = []
        self.generate_robber_states()
        self.n_robber_states = len(self.robber_states)

        self.police_epsilon = p_epsilon
        self.police_states = []
        self.generate_police_states()
        self.n_police_states = len(self.police_states)

        self.states = []
        self.states_matrix = np.full((self.n_robber_states, self.n_police_states), State)
        self.generate_states()

    def __getitem__(self, i):
        return self.states[i]

    def __len__(self):
        return len(self.states)

    def generate_states(self):
        state_count = count(0)
        for robber in self.robber_states:
            for police in self.police_states:
                state = State(id=next(state_count),
                              robber=copy.deepcopy(robber),
                              police=copy.deepcopy(police))
                if robber.position == self.robber_initial_position and police.position == self.police_initial_position:
                    self.initial_state = state

                self.states.append(state)
                self.states_matrix[robber.id, police.id] = state

        for state in self.states:
            state.robber.initialize_q_values()
            state.police.initialize_q_values()

    def generate_robber_states(self):
        robber_count = count(0)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.robber_states.append(Robber(id=next(robber_count),
                                                 position=(i, j),
                                                 bank_position=self.bank_position,
                                                 epsilon=self.robber_epsilon))

        for state in self.robber_states:
            state.neighbours.append(state)  # can stand still
            for other_state in self.robber_states:
                if np.abs(state.position[0]-other_state.position[0]) == 1 and np.abs(state.position[1]-other_state.position[1]) == 0:
                    state.neighbours.append(other_state)    # column neighbour
                elif np.abs(state.position[0]-other_state.position[0]) == 0 and np.abs(state.position[1]-other_state.position[1]) == 1:
                    state.neighbours.append(other_state)    # row neighbour

    def generate_police_states(self):
        police_count = count(0)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.police_states.append(Police(id=next(police_count), position=(i, j),
                                                 epsilon=self.robber_epsilon))

        for state in self.police_states:
            state.neighbours.append(state)  # can stand still
            for other_state in self.police_states:
                if np.abs(state.position[0]-other_state.position[0]) == 1 and np.abs(state.position[1]-other_state.position[1]) == 0:
                    state.neighbours.append(other_state)    # column neighbour
                elif np.abs(state.position[0]-other_state.position[0]) == 0 and np.abs(state.position[1]-other_state.position[1]) == 1:
                    state.neighbours.append(other_state)    # row neighbour

    def state_where(self, robber=None, police=None):
        return self.states_matrix[robber.id, police.id]

    def subset_where(self, robber_id=None, police_id=None):
        if police_id is None and robber_id in range(self.n_robber_states):
            return list(self.states_matrix[robber_id, :])

        elif robber_id is None and police_id in range(self.n_police_states):
            return list(self.states_matrix[:, police_id])

        else:
            print('Error: Specify a valid id for a robber or a police state.')
            return None

    def possible_next_states(self, state, action):
        next_robber_state = state.robber.neighbours[action]
        subset = self.subset_where(robber_id=next_robber_state.id)
        return [subset[next_police.id] for next_police in state.police.neighbours]

    def reset_actions(self):
        for state in self.states:
            state.robber.action = 0


class RobberReward:
    def __init__(self):
        self.robbing_bank = 1
        self.being_caught = -2

    def __call__(self, state):
        if state.robber_caught():
            return self.being_caught
        elif state.robber_robbing():
            return self.robbing_bank
        else:
            return 0


class PoliceReward:
    def __init__(self):
        self.bank_robbed = -1
        self.catching_robber = 2

    def __call__(self, state):
        if state.robber_robbing():
            return self.bank_robbed
        elif state.robber_caught():
            return self.catching_robber
        else:
            return 0


class State:
    def __init__(self, id, robber, police):
        self.id = id
        self.robber = robber
        self.police = police

    def __str__(self):
        return str(self.robber) + '\n' + str(self.police)

    def robber_caught(self):
        if self.robber.position == self.police.position:
            return True
        else:
            return False

    def robber_robbing(self):
        if self.robber.is_in_bank() and not self.robber_caught():
            return True
        else:
            return False

#class Agent:
#    def __init__(self, id, position, bank_position):


class Robber:   # (Agent)
    def __init__(self, id, position, bank_position, epsilon):
        self.id = id
        self.position = position
        self.epsilon = epsilon
        self.neighbours = []
        self.action = 0
        self.bank_position = bank_position

        self.q_a = []

    def __str__(self):
        return 'robber @ ' + str(self.position)

    def initialize_q_values(self):
        for _ in self.neighbours:
            q = QValue()
            self.q_a.append(q)

    def is_in_bank(self):
        return self.position == self.bank_position

    def select_action(self):
        if choice(a=[True, False], size=1, p=[1-self.epsilon, self.epsilon]):     # greedy
            return self.action
        else:
            actions = list(np.r_[:len(self.neighbours)])
            actions.remove(self.action)
            return choice(actions)


class Police:   # (Agent)
    def __init__(self, id, position, epsilon):
        self.id = id
        self.epsilon = epsilon
        self.position = position
        self.neighbours = []
        self.action = 0
        self.q_a = []

    def __str__(self):
        return 'police @ ' + str(self.position)

    def initialize_q_values(self):
        for _ in self.neighbours:
            q = QValue()
            self.q_a.append(q)

    def select_action(self):
        if choice(a=[True, False], size=1, p=[1-self.epsilon, self.epsilon]):     # greedy
            return self.action
        else:
            actions = list(np.r_[:len(self.neighbours)])
            actions.remove(self.action)
            return choice(actions)


class QValue:   # action-state value
    def __init__(self):
        self.value = 0.0
        self.n_updates = 1

    def __call__(self):
        return self.value

    def __str__(self):
        return '{:.4f}'.format(self.value)

    def update_value(self, correction):
        self.value += self.n_updates ** (-2/3) * correction
        self.n_updates += 1

