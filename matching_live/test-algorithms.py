import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from algorithms import MatchingPennies2, BlockFlipperWithExtension
    import matplotlib.pyplot as plt 
    import sciplotlib.style as splstyle


    return BlockFlipperWithExtension, MatchingPennies2, mo, np, plt, splstyle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create some sequences of choices and look at the internals of MatchingPennies2
    """)
    return


@app.cell
def _(np, plt):
    def plot_subject_and_computer_choice(computer_choice, subject_choice=None, subject_reward=None,
                                        fig=None, axs=None):

        if (fig is None) and (axs is None):
            fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
            fig.set_size_inches(6, 3)

        num_trials = len(computer_choice)
        trial_indices = np.arange(num_trials)

        if subject_choice is not None: 
            if subject_reward is None: 
                axs[0].scatter(trial_indices, subject_choice, lw=0, color='black')
            else:
                rewarded_trials = np.where(subject_reward > 0)[0]
                unrewarded_trials = np.where(subject_reward == 0)[0]
                axs[0].scatter(trial_indices[rewarded_trials], subject_choice[rewarded_trials], lw=0, color='red')
                axs[0].scatter(trial_indices[unrewarded_trials], subject_choice[unrewarded_trials], lw=0, color='black')

        if subject_reward is None: 
            axs[1].scatter(trial_indices, computer_choice)
        else:
            cpu_reward = (1 - subject_reward)
            rewarded_trials = np.where(cpu_reward > 0)[0]
            unrewarded_trials = np.where(cpu_reward == 0)[0]
            axs[1].scatter(trial_indices[rewarded_trials], computer_choice[rewarded_trials], lw=0, color='red')
            axs[1].scatter(trial_indices[unrewarded_trials], computer_choice[unrewarded_trials], lw=0, color='black')

        axs[0].set_title('Subject choice', size=11)
        axs[1].set_title('Comptuer choice', size=11)

        print('Computer P(right): %.2f' % np.mean(computer_choice)) 

        return fig, axs

    return (plot_subject_and_computer_choice,)


@app.cell
def _(MatchingPennies2):
    mp_algo = MatchingPennies2(N=4, alpha=0.05, invert_prediction=False)
    return (mp_algo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test 1: Just sample without any history and make sure it does random left/right choices
    """)
    return


@app.cell
def _(mp_algo, np, plot_subject_and_computer_choice, plt, splstyle):

    num_trials = 10000
    computer_choice = np.zeros((num_trials, )) + np.nan
    for trial_idx in np.arange(num_trials):
        computer_choice[trial_idx] = (mp_algo.sample() == 'R')

    with plt.style.context(splstyle.get_style('nature-reviews')): 
        _fig, _axs = plot_subject_and_computer_choice(computer_choice=computer_choice, subject_choice=None)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test 2, see how quickly the computer catches on when subject choose a single choice
    """)
    return


@app.cell
def _(MatchingPennies2, np, plot_subject_and_computer_choice, plt, splstyle):
    def test_repeated_choices(repeated_choice=1, num_trials=100):
    
        mp_algo = MatchingPennies2(N=4, alpha=0.05, invert_prediction=False)
        computer_choice = np.zeros((num_trials, )) + np.nan
        subject_choice = np.repeat(repeated_choice, num_trials)  # 0 : 'L' 1 : 'R'
        subject_reward = np.zeros((num_trials, )) + np.nan
        for trial_idx in np.arange(num_trials):
            computer_choice[trial_idx] = (mp_algo.sample() == 'R')
    
            if subject_choice[trial_idx] == 1:
                subject_choice_str = 'R'
            else:
                subject_choice_str = 'L'

            if subject_choice[trial_idx] == computer_choice[trial_idx]:
                reward = 1
            else:
                reward = 0

            # NOTE: we may want to do the same where the computer still chooses randomly 
            # on the first 4 trials... (but note, still updates!)
            mp_algo.update(subject_choice_str, reward)
            subject_reward[trial_idx] = reward
        return computer_choice, subject_choice, subject_reward

    _cpu_choice, _s_choice, _s_reward = test_repeated_choices()
    with plt.style.context(splstyle.get_style('nature-reviews')): 
        _fig, _axs = plot_subject_and_computer_choice(computer_choice=_cpu_choice, subject_choice=_s_choice, subject_reward=_s_reward)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Try a win-stay lose-switch strategy
    """)
    return


@app.function
def win_stay_lose_switch(choice, reward):

    if reward == 1: 
        next_choice = choice
    else: 
        next_choice = 1 - choice
            
    return next_choice


@app.cell
def _(MatchingPennies2, np, plot_subject_and_computer_choice, plt, splstyle):
    def test_wsls_choices(num_trials=100):
    
        mp_algo = MatchingPennies2(N=4, alpha=0.05, invert_prediction=False)
        computer_choice = np.zeros((num_trials, )) + np.nan
        subject_choice = np.zeros((num_trials, )) + np.nan
        subject_reward = np.zeros((num_trials, )) + np.nan
        for trial_idx in np.arange(num_trials-1):
            computer_choice[trial_idx] = (mp_algo.sample() == 'R')

            if trial_idx == 0: 
                subject_choice[trial_idx] = np.random.choice([0, 1])
    
            if subject_choice[trial_idx] == 1:
                subject_choice_str = 'R'
            else:
                subject_choice_str = 'L'

            if subject_choice[trial_idx] == computer_choice[trial_idx]:
                reward = 1
            else:
                reward = 0

            # NOTE: we may want to do the same where the computer still chooses randomly 
            # on the first 4 trials... (but note, still updates!)
            mp_algo.update(subject_choice_str, reward)

            subject_reward[trial_idx] = reward 
            # subject choose the next choice based on previous choice and reward 
            subject_choice[trial_idx+1] = win_stay_lose_switch(subject_choice[trial_idx], reward)
        

        return computer_choice, subject_choice, subject_reward

    _cpu_choice, _s_choice, _s_reward = test_wsls_choices()
    with plt.style.context(splstyle.get_style('nature-reviews')): 
        _fig, _axs = plot_subject_and_computer_choice(computer_choice=_cpu_choice, subject_choice=_s_choice, subject_reward=_s_reward)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test a sequence that the computer should be able to spot
    """)
    return


@app.cell
def _(MatchingPennies2, np, plot_subject_and_computer_choice, plt, splstyle):
    def test_repeated_sequence(sequence_length=4, num_trials=100):
    
        mp_algo = MatchingPennies2(N=4, alpha=0.05, invert_prediction=False)
        computer_choice = np.zeros((num_trials, )) + np.nan
        sequence = np.random.choice([0, 1], sequence_length)
        subject_choice = np.tile(sequence, int(np.ceil(num_trials / sequence_length)))[0:num_trials]
        print('Sequence')
        print(sequence)
        print('First 20 trials')
        print(subject_choice[0:20])
        subject_reward = np.zeros((num_trials, )) + np.nan
        for trial_idx in np.arange(num_trials):
            computer_choice[trial_idx] = (mp_algo.sample() == 'R')
    
            if subject_choice[trial_idx] == 1:
                subject_choice_str = 'R'
            else:
                subject_choice_str = 'L'

            if subject_choice[trial_idx] == computer_choice[trial_idx]:
                reward = 1
            else:
                reward = 0

            # NOTE: we may want to do the same where the computer still chooses randomly 
            # on the first 4 trials... (but note, still updates!)
            mp_algo.update(subject_choice_str, reward)
            subject_reward[trial_idx] = reward
        return computer_choice, subject_choice, subject_reward

    _cpu_choice, _s_choice, _s_reward = test_repeated_sequence()
    with plt.style.context(splstyle.get_style('nature-reviews')): 
        _fig, _axs = plot_subject_and_computer_choice(computer_choice=_cpu_choice, subject_choice=_s_choice, subject_reward=_s_reward)
        plt.show()
    return (test_repeated_sequence,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test a long pattern, which the computer should not be able to catch on to
    """)
    return


@app.cell
def _(plot_subject_and_computer_choice, plt, splstyle, test_repeated_sequence):
    _cpu_choice, _s_choice, _s_reward = test_repeated_sequence(sequence_length=3, num_trials=200)
    with plt.style.context(splstyle.get_style('nature-reviews')): 
        _fig, _axs = plot_subject_and_computer_choice(computer_choice=_cpu_choice, subject_choice=_s_choice, subject_reward=_s_reward)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Do the simulation with different sequence lenghts many times, just to check that by the end the computer always win for sequence 4 or shorter, but not for all 5 sequences or greater
    """)
    return


@app.cell
def _(np, test_repeated_sequence):
    def do_sequence_length_simulations(sequence_lengths_to_test = np.arange(1, 10)): 
        num_simulations = 100 
        num_trials = 100
        num_sequences =len(sequence_lengths_to_test)

        computer_rewards_per_sim_and_seq = np.zeros((num_sequences, num_simulations, num_trials)) + np.nan

        for seq_idx, seq_length in enumerate(sequence_lengths_to_test): 
            for sim_idx in np.arange(num_simulations):
                _, _, _s_reward = test_repeated_sequence(sequence_length=seq_length, num_trials=num_trials)
                computer_rewards_per_sim_and_seq[seq_idx, sim_idx, :] = 1 - _s_reward

        return computer_rewards_per_sim_and_seq 

    return (do_sequence_length_simulations,)


@app.cell
def _(do_sequence_length_simulations, np):
    computer_rewards_per_sim_and_seq = do_sequence_length_simulations(sequence_lengths_to_test = np.arange(1, 10))
    sequence_lengths_to_test = np.arange(1, 10)
    return computer_rewards_per_sim_and_seq, sequence_lengths_to_test


@app.cell
def _(
    computer_rewards_per_sim_and_seq,
    np,
    plt,
    sequence_lengths_to_test,
    splstyle,
):
    def plot_sequence_simulations(computer_rewards_per_sim_and_seq, sequence_lengths_to_test,
                                  fig=None, ax=None):

        if (fig is None) and (ax is None):
            fig, ax = plt.subplots( )

        for seq_idx, seq_length in enumerate(sequence_lengths_to_test): 

            ax.plot(np.mean(computer_rewards_per_sim_and_seq[seq_idx, :, :], axis=0),
                   label=seq_length)

        ax.legend() 
    
        return fig, ax

    with plt.style.context(splstyle.get_style('nature-reviews')): 

        _fig, _ax = plot_sequence_simulations(computer_rewards_per_sim_and_seq, sequence_lengths_to_test)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test the two-armed bandit opponent
    """)
    return


@app.cell
def _(np, plt):
    def plot_2ab(subject_choice, subject_reward, fig=None, ax=None):

        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
            fig.set_size_inches(6, 1.5)

        num_trials = len(subject_choice)

        rewarded_trials = np.where(subject_reward == 1)[0]
        not_rewarded_trials = np.where(subject_reward == 0)[0]

        ax.scatter(not_rewarded_trials, subject_choice[not_rewarded_trials], color='black')
        ax.scatter(rewarded_trials, subject_choice[rewarded_trials], color='red')

        ax.set_ylim([-0.05, 1.05])

        return fig, ax



    return (plot_2ab,)


@app.cell
def _(BlockFlipperWithExtension, np, plot_2ab, plt, splstyle):
    def test_2ab_repeated_choice(num_trials=100):
        bandit_ev = BlockFlipperWithExtension(
                p_high=0.6,
                p_low=0.0,
                lambda_=25.0,
                extend_block=5,
                block_extend_threshold=0.2,
            )

        subject_choice = np.zeros((num_trials, )) + np.nan
        subject_reward = np.zeros((num_trials, )) + np.nan
    
        for trial_idx in np.arange(num_trials): 
            choice = 'L'
            reward_bin = bandit_ev.trial(choice)

            subject_choice[trial_idx] = (choice == 'R') + 0.0
            subject_reward[trial_idx] = reward_bin

        return subject_choice, subject_reward

    _s_choice, _s_reward = test_2ab_repeated_choice(num_trials=100)

    with plt.style.context(splstyle.get_style('nature-reviews')): 
        _fig, _ax = plot_2ab(_s_choice, _s_reward)

        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test a win stay lose switch strategy
    """)
    return


@app.cell
def _(BlockFlipperWithExtension, np, plot_2ab, plt, splstyle):
    def test_2ab_wsls(num_trials=100):
        bandit_ev = BlockFlipperWithExtension(
                p_high=0.6,
                p_low=0.0,
                lambda_=25.0,
                extend_block=5,
                block_extend_threshold=0.2,
            )

        subject_choice = np.zeros((num_trials, )) + np.nan
        subject_reward = np.zeros((num_trials, )) + np.nan
    
        for trial_idx in np.arange(num_trials): 
            if trial_idx == 0:
                choice = np.random.choice(['L', 'R'])
            else:
                if subject_reward[trial_idx-1] == 1: 
                    if subject_choice[trial_idx-1] == 1: 
                        choice = 'R'
                    else:
                        choice = 'L'
                else:
                    if subject_choice[trial_idx-1] == 1: 
                        choice = 'L'
                    else:
                        choice = 'R'
        
            reward_bin = bandit_ev.trial(choice)

            subject_choice[trial_idx] = (choice == 'R') + 0.0
            subject_reward[trial_idx] = reward_bin

        return subject_choice, subject_reward

    _s_choice, _s_reward = test_2ab_wsls(num_trials=100)

    with plt.style.context(splstyle.get_style('nature-reviews')): 
        _fig, _ax = plot_2ab(_s_choice, _s_reward)

        plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
