from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product


class TicTacToeE():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix
        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        idx_to_check = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for idx in idx_to_check:
            if (curr_state[idx[0]] + curr_state[idx[1]] + curr_state[idx[2]]) == 15:
                return True
        return False
 

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up
        if self.is_winning(curr_state) == True:
            return True, 'Win'
        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'
        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""
        agent_values, env_values = self.allowed_values(curr_state)
        allowed_positions = self.allowed_positions(curr_state)
        agent_actions = product(allowed_positions, agent_values)
        env_actions = product(allowed_positions, env_values)
        return (agent_actions, env_actions)

    def reward(self, curr_state, by_agent):
        """The Reward if action taken leads to the curr_state. Rewards will be different
        depending on whether the action is taken by the agent or the environment.
        """
        is_terminal, result = self.is_terminal(curr_state)
        r = 0
        if result=='Win':
            r = 10 if by_agent else -10
        elif is_terminal:
            r = 0
        else:
            r = -1 if by_agent else 0
        return r, is_terminal

    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        curr_state[curr_action[0]] = curr_action[1]
        return curr_state


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board
        position after agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        assert curr_action in list(self.action_space(curr_state)[0]) # Agent should be allowed to take the intended action.
        state = curr_state.copy()    # Creating copy as don't want to write on the passed object
        next_state = self.state_transition(state, curr_action) # Move to next state.
        r, is_terminal = self.reward(next_state, by_agent=True)
        # Environment's Move if agents move didn't lead to terminal state, choose action randomly from env's available action_space
        if not is_terminal:
            env_actions = list(self.action_space(next_state)[1])
            next_state = self.state_transition(next_state, random.choice(env_actions))
            er, is_terminal = self.reward(next_state, by_agent=False)
            r += er
        return (next_state, r, is_terminal)

    def reset(self):
        return self.state