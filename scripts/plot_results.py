

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('dark_background')


def plot_model_comparison(preds_dir):
    # aggregate evaluation.csv files
    dfs = []
    for eval_csv in preds_dir.glob('*/evaluation.csv'):
        df = pd.read_csv(eval_csv)
        model_name = eval_csv.parent.name
        df['Model'] = model_name
        dfs.append(df)
    df = pd.concat(dfs)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    sns.set()
    plt.style.use('dark_background')
    
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xticks([0.25, 0.5, 1.0, 2, 4, 8])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.spines['left'].set_position(('data', 1))
        ax.yaxis.tick_left()
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1])
        ax.set_ylim(0.5, 1)
        ax.tick_params(axis='y', colors='white')
        ax.grid(False)

    # plot in NATIVE referential
    _ = sns.lineplot(
        data=df, x="Resize Factor", y="Dice Axon Native", palette='rocket',
        hue='Model', ax=axs[0], marker='.'
    )
    _ = sns.lineplot(
        data=df, x="Resize Factor", y="Dice Myelin Native", palette='rocket',
        hue='Model', ax=axs[1], marker='.', legend=False
    )
    sns.despine()
    axs[0].legend()
    box0 = axs[0].get_position()
    axs[0].set_position([box0.x0, box0.y0 + box0.height * 0.1, box0.width, box0.height * 0.9])
    axs[0].legend(loc='upper center', bbox_to_anchor=(1, -0.1), ncols=2)
    axs[0].set_ylabel("")
    axs[0].set_title("Axon Dice in Native Referential")
    box1 = axs[1].get_position()
    axs[1].set_position([box1.x0, box1.y0 + box1.height * 0.1, box1.width, box1.height * 0.9])
    # axs[1].legend()
    # axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncols=2)
    axs[1].set_ylabel("")
    axs[1].set_title("Myelin Dice in Native Referential")

    plt.style.use('dark_background')
    plt.savefig(preds_dir / 'model_comparison.png', dpi=200)

def plot_and_write_results(results_dir):
    assert (results_dir / 'evaluation.csv').exists(), f'No evaluation.csv found in {results_dir}'
    df = pd.read_csv(results_dir / 'evaluation.csv')
    fig, axs = plt.subplots(2, 1, figsize=(8, 14))
    sns.set()
    plt.style.use('dark_background')
    
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xticks([0.25, 0.5, 1.0, 2, 4, 8])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.spines['left'].set_position(('data', 1))
        ax.yaxis.tick_left()
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1])
        ax.set_ylim(0.55, 1)
        ax.tick_params(axis='y', colors='white')
        ax.grid(False)

    # plot in SHIFTED referential
    _ = sns.lineplot(
        data=df, x="Resize Factor", y="Dice Axon Resized",
        ax=axs[0], color='limegreen', label='axon', marker='.'
    )
    _ = sns.lineplot(
        data=df, x="Resize Factor", y="Dice Myelin Resized",
        ax=axs[0], color='springgreen', label="myelin", marker='.'
    )
    _ = sns.lineplot(
        data=df, x="Resize Factor", y="Dice Axon Interpolation",
        ax=axs[0], color='limegreen', label="axon gt interpolation", 
        linestyle='--', alpha=0.5
    )
    _ = sns.lineplot(
        data=df, x="Resize Factor", y="Dice Myelin Interpolation",
        ax=axs[0], color='springgreen', label="myelin gt interpolation", 
        linestyle='--', alpha=0.5
    )
    sns.despine()
    axs[0].legend()
    axs[0].set_ylabel("")
    axs[0].set_title("Dice - Shifted Referential")

    # plot in NATIVE referential
    _ = sns.lineplot(
        data=df, x="Resize Factor", y="Dice Axon Native",
        ax=axs[1], color='fuchsia', label='axon', marker='.'
    )
    _ = sns.lineplot(
        data=df, x="Resize Factor", y="Dice Myelin Native",
        ax=axs[1], color='mediumvioletred', label="myelin", marker='.'
    )
    _ = sns.lineplot(
        data=df, x="Resize Factor", y="Dice Axon Interpolation",
        ax=axs[1], color='fuchsia', label="axon gt interpolation", 
        linestyle='--', alpha=0.5
    )
    _ = sns.lineplot(
        data=df, x="Resize Factor", y="Dice Myelin Interpolation",
        ax=axs[1], color='mediumvioletred', label="myelin gt interpolation", 
        linestyle='--', alpha=0.5
    )
    sns.despine()
    axs[1].legend()
    axs[1].set_ylabel("")
    axs[1].set_title("Dice - Native Referential")

    plt.style.use('dark_background')
    plt.savefig(results_dir / 'evaluation.png', dpi=200)