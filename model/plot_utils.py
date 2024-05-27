import colorsys
import matplotlib.pyplot as plt
import numpy as np

def get_spaced_colors(n):
    '''
    Helper function that assigns somewhat unique colors to agents
    '''
    colors = np.zeros((n, 3))
    idxs = np.arange(0, n, 1).astype(int)
    np.random.shuffle(idxs)
    j = 0
    for i in idxs:
        if i == 0:
            colors[i] = [0, 0, 0]
        else:
            h = j * 1.0 / n
            rgb = colorsys.hsv_to_rgb(h, 1, 1)
            colors[i] = rgb
            j += 1
    return colors


def plot_traj(list, best_action):
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)

    axs = axs.ravel()
    for j, t in enumerate(list):
        hist, future, r_goal, robot_idx = t
        current_hist = hist[:, 0:6].cpu().numpy()
        smapled_robots_state = future[:, robot_idx].cpu().numpy().transpose(1, 0, 2)
        human_future = future.cpu().numpy().transpose(1, 0, 2)
        human_future = human_future.reshape(100, 6, 12, 2)
        human_future = human_future.mean(axis=0)
        ax = axs[j]
        # fig, ax = plt.subplots()

        for k, traj in enumerate(smapled_robots_state):
            if k == 0:
                label = 'CEM Sample'
            else:
                label = '_nolegend_'
            ax.plot(traj[:, 0], traj[:, 1], 'tab:green', linewidth=3, label=label, alpha=.2)

        for i in range(0, 6):
            if i == 0:
                label_h = 'History'
            else:
                label_h = '_nolegend_'

            if i == 0:
                color = 'black'

            else:
                color = 'tab:grey'
                if i == 1:
                    label_hp = "Mean Forecast"
                else:
                    label_hp = '_nolegend_'

                ax.plot(human_future[i, :, 0], human_future[i, :, 1], 'tab:orange', linewidth=3,
                        label=label_hp, marker='o', alpha=1.)
            current_hist_i = current_hist[:, i]
            current_hist_i = current_hist_i[current_hist_i != 0].reshape(-1, 2)
            observed_line = ax.plot(current_hist_i[:, 0], current_hist_i[:, 1],
                                    color, linewidth=3, label=label_h, marker='^', alpha=.5, )[0]

            # observed_line.axes.annotate(
            #     "",
            #     xytext=(
            #         current_hist[-1, i, 0],
            #         current_hist[-1, i, 1],
            #     ),
            #     xy=(
            #         human_future[i, 0, 0],
            #         human_future[i, 0, 1],
            #     ),
            #     arrowprops=dict(
            #         arrowstyle="-|>", color=color, lw=5, alpha=1.
            #     ),
            #     size=15, alpha=.2
            # )

        ax.set_title('CEM-iteration: ' + str(j))
        ax.plot(r_goal[0, 0], r_goal[0, 1], 'black', marker="X", markersize=20, label='Goal', linestyle='None')
        # ax.set_aspect("equal")

    best_action_abs = np.cumsum(best_action, axis=0) + current_hist[-1, 0]
    ax.plot(best_action_abs[:, 0], best_action_abs[:, 1], 'black', linewidth=3, label='Mean Robot Plan', marker='s',
            alpha=1.)
    # ax.legend(loc="upper right")
    ax.legend()
    # ax.yaxis.set_units('m')
    # ax.xaxis.set_units('m')
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)
    # plt.tight_layout()
    plt.show()

    def plot_trajectory(self, trajectories, goals):
        plt.figure()

        for i in range(1*6):
             # Plot trajectory
            xs, ys = trajectories[:, i, 0], trajectories[:, i, 1]
            plt.plot(xs, ys, label=f'Trajectory {i + 1}')

    # Plot corresponding goal
            goal = goals[i]
            if i < 6:
                plt.plot(*goal, 'ro')  # 'ro' plots a red circle
                plt.text(goal[0], goal[1], f'Goal {i + 1}', fontsize=12, ha='right')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Trajectories and their Goals')
        # plt.legend()
        plt.grid(True)
        plt.show()