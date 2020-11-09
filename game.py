from classes import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as matplotlib_colormap
from matplotlib.animation import FuncAnimation
import sys


def run_game(states, ax, t):
    max_game_time = 100
    ax.clear()
    # ax.set_title(str(t) + " time steps")
    grid = np.ones(states.grid_size)
    grid[states.bank_position] = 0
    colormap = matplotlib_colormap.get_cmap('RdBu')
    ax.matshow(grid, cmap=colormap)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    state = states.initial_state
    (r2, r1) = state.robber.position
    (p2, p1) = state.police.position
    robber_mark, = ax.plot(r1, r2, color=colormap(0.15), marker='$R$', markersize=16)     # create mark for robber
    police_mark, = ax.plot(p1, p2, color=colormap(0.85), marker='$P$', markersize=16)     # create mark for police
    plt.pause(0.5)                          # pause and draw
    i = 0

    save_game = []
    save_game.append(dict(robber=(r1, r2), police=(p1, p2)))

    while i < max_game_time:
        i += 1
        state = states.state_where(robber=state.robber.neighbours[state.robber.select_action()],
                                   police=state.police.neighbours[state.police.select_action()])

        (r2, r1) = state.robber.position
        (p2, p1) = state.police.position
        robber_mark.set_data(r1, r2)
        police_mark.set_data(p1, p2)
        save_game.append(dict(robber=(r1, r2), police=(p1, p2)))

        plt.pause(0.2)  # pause and update (draw) positions

    return save_game, robber_mark, police_mark, max_game_time



def run_game_scores(states, t):
    fig, ax = plt.subplots(2, 1)
    fig.set_tight_layout(True)

    max_game_time = 100
    # ax[0].clear()
    # ax.set_title(str(t) + " time steps")
    grid = np.ones(states.grid_size)
    grid[states.bank_position] = 0
    colormap = matplotlib_colormap.get_cmap('RdBu')
    ax[0].matshow(grid, cmap=colormap)
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    state = states.initial_state

    ##
    time_line = np.arange(max_game_time+1)
    robber_score = robber_rewards(state=state)
    police_score = police_rewards(state=state)
    robber_score_all = np.full(shape=max_game_time+1, fill_value=np.nan)
    police_score_all = np.full(shape=max_game_time+1, fill_value=np.nan)
    robber_score_all[0] = robber_score
    police_score_all[0] = police_score
    robber_score_plot, = ax[1].plot(time_line, robber_score_all, '-', color=colormap(0.15))
    police_score_plot, = ax[1].plot(time_line, police_score_all, '-', color=colormap(0.85))
    ax[1].set_xlim(xmin=0, xmax=max_game_time)
    ax[1].set_ylim(ymin=-10, ymax=10)
    ##

    (r2, r1) = state.robber.position
    (p2, p1) = state.police.position

    ##
    robber_mark, = ax[0].plot(r1, r2, color=colormap(0.15), marker='$R$', markersize=16)  # create mark for robber
    police_mark, = ax[0].plot(p1, p2, color=colormap(0.85), marker='$P$', markersize=16)  # create mark for police
    ##

    # robber_mark, = ax.plot(r1, r2, color=colormap(0.15), marker='$R$', markersize=16)     # create mark for robber
    # police_mark, = ax.plot(p1, p2, color=colormap(0.85), marker='$P$', markersize=16)     # create mark for police
    plt.pause(0.5)                          # pause and draw
    i = 0

    save_game = []
    save_game.append(dict(robber=(r1, r2), police=(p1, p2)))

    while i < max_game_time:
        i += 1
        state = states.state_where(robber=state.robber.neighbours[state.robber.select_action()],
                                   police=state.police.neighbours[state.police.select_action()])

        ##
        # robber_score += robber_rewards(state=state)
        # police_score += police_rewards(state=state)
        robber_score_all[i] = robber_score_all[i-1] + robber_rewards(state=state)
        print(robber_score_all)
        police_score_all[i] = police_score_all[i-1] + police_rewards(state=state)
        robber_score_plot.set_data(time_line, robber_score_all)
        police_score_plot.set_data(time_line, police_score_all)
        ##

        (r2, r1) = state.robber.position
        (p2, p1) = state.police.position
        robber_mark.set_data(r1, r2)
        police_mark.set_data(p1, p2)
        save_game.append(dict(robber=(r1, r2), police=(p1, p2)))

        plt.pause(0.2)  # pause and update (draw) positions

    return save_game, robber_mark, police_mark, max_game_time


def update(i):
    (r1, r2) = replay[i]["robber"]
    (p1, p2) = replay[i]["police"]
    robber_mark.set_data(r1, r2)
    police_mark.set_data(p1, p2)


def sarsa(agent, action, reward, next_agent, next_action, gamma=0.8):
    qt = agent.q_a[action]  # get the q-value associated to this action
    qt_next = next_agent.q_a[next_action]   # get the q-value associated to the next action
    correction = reward + gamma * qt_next() - qt()  # compare the actual reward with the expected reward
    qt.update_value(correction=correction)  # update the current q-value
    q_values = [qsa() for qsa in agent.q_a]     # get q-values of the agent
    agent.action = np.argmax(q_values)  # choose action based on maximal expected reward


def time_step(states, state, robber_rewards, police_rewards):

    robber_reward = robber_rewards(state=state)
    police_reward = police_rewards(state=state)

    robber = state.robber
    police = state.police
    robber_action = robber.select_action()
    police_action = police.select_action()

    next_state = states.state_where(robber=robber.neighbours[robber_action],
                                    police=police.neighbours[police_action])
    next_robber = next_state.robber
    next_police = next_state.police

    sarsa(agent=robber, action=robber_action, reward=robber_reward, next_agent=next_robber,
          next_action=next_robber.action)
    sarsa(agent=police, action=police_action, reward=police_reward, next_agent=next_police,
          next_action=next_police.action)

    state = next_state

    return state


if __name__ == '__main__':

    grid_size = (4, 4)
    robber_position = (0, 0)
    police_position = (3, 3)
    bank_position = (1, 1)
    greedy_robber = 0.3
    greedy_police = 0.05

    states = StateSpace(grid_size=grid_size, r=robber_position, p=police_position, b=bank_position,
                        r_epsilon=greedy_robber, p_epsilon=greedy_police)

    robber_rewards = RobberReward()
    police_rewards = PoliceReward()

    state = states.initial_state
    n = 1000003

    for t in range(n):
        # compute next step
        state = time_step(states=states, state=state, robber_rewards=robber_rewards, police_rewards=police_rewards)

        if t in [100000, 200000, 500000, 1000000]:
            fig, ax = plt.subplots()
            fig.set_tight_layout(True)

            print(t)

            replay, robber_mark, police_mark, max_game_time = run_game(states=states, ax=ax, t=t)

            # FuncAnimation will call the 'update' function for each frame; here
            # animating over 10 frames, with an interval of 200ms between frames.
            # anim = FuncAnimation(fig, update, frames=max_game_time, interval=300)

            # anim.save('game-' + str(t) + '.gif', dpi=100)
