import polars as pl
import matplotlib.pyplot as plt

for i in range(20):
    Base = pl.read_csv(f"Assignment 2/20Runs/Results/PreyPredator_Grass{i}.csv")

    BasePredator = Base.filter(pl.col("AID") == "Predator")
    BasePrey = Base.filter(pl.col("AID") == "Prey")

    BasePredator = BasePredator.with_columns(
        (BasePredator["AgentCount"]/10).alias("Proportion of Predators"),
        (BasePredator["frame"]/60).alias("Time (seconds)")
    )

    BasePrey = BasePrey.with_columns(
        (BasePrey["AgentCount"]/100).alias("Proportion of Prey"),
        (BasePrey["frame"]/60).alias("Time (seconds)")
    )

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(BasePredator["Time (seconds)"], BasePredator["Proportion of Predators"], 'g-')
    ax1.set_ylim(0,16)
    ax2.plot(BasePrey["Time (seconds)"], BasePrey["Proportion of Prey"], 'b-')
    ax2.set_ylim(0,16)

    ax1.set_xlabel('Time (Simulated Seconds)', fontsize=16)
    ax1.set_ylabel('Proportion of Predators', color='g', fontsize=16)
    ax2.set_ylabel('Proportion of Prey', color='b', fontsize=16)

    plt.title("Centered Patch of Grass", fontsize=16)
    plt.xlim(0, 35000/60)
    plt.savefig(f"Assignment 2/20Runs/Figures/Grass{i}.png")