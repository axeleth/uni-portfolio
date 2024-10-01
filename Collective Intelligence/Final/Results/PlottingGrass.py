from enum import Enum, auto
import sys
import pygame as pg
from pygame.math import Vector2
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize
import vi
from numpy import random as rnd
import polars as pl
import seaborn as sns
import numpy as np
import math
import matplotlib.pyplot as plt

Base = pl.read_csv("Assignment 2/Results/PreyPredator_Grass.csv")

BasePredator = Base.filter(pl.col("AID") == "Predator")
BasePrey = Base.filter(pl.col("AID") == "Prey")

print(BasePredator)
print(BasePrey)

BasePredator = BasePredator.with_columns(
    (BasePredator["AgentCount"]/10).alias("Proportion of Predators"),
    (BasePredator["frame"]/60).alias("Time (seconds)")
)


BasePrey = BasePrey.with_columns(
    (BasePrey["AgentCount"]/100).alias("Proportion of Prey"),
    (BasePrey["frame"]/60).alias("Time (seconds)")
)

print(BasePredator)
print(BasePrey)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(BasePredator["Time (seconds)"], BasePredator["Proportion of Predators"], 'g-')
ax1.set_ylim(0,40)
ax2.plot(BasePrey["Time (seconds)"], BasePrey["Proportion of Prey"], 'b-')
ax2.set_ylim(0,40)

ax1.set_xlabel('Time (frames)')
ax1.set_ylabel('Proportion of Predators', color='g')
ax2.set_ylabel('Proportion of Prey', color='b')

plt.title("Only Energy Implemented")
plt.xlim(0, 25000/60)
plt.show()