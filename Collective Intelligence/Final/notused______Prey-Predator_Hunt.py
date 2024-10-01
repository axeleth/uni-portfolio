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

EPS = sys.float_info.epsilon
SPEEDFACTOR = 1
SECONDS = 300
rnd.seed(5)

class EcosystemConfig(Config):
    #RabbitConfig
    prey_movement_speed = 6
    Prey_Reproduction_Rate = 0.0018
    
    #FoxConfig
    predator_movement_speed = 25
    death_Rate = 0.0005
    default_energy_level = 2.7 * 60
    Predator_Reproduction_Rate = 0.07
    eating_distance = 16

    #General Config
    dT = 0.1


class Predator(Agent):
    config: EcosystemConfig

    def on_spawn(self):
        self.move = Vector2(rnd.uniform(-1,1),rnd.uniform(-1,1))
        self.agent_id = "Predator"
        self.energy = self.config.default_energy_level

    def update(self):
        self.save_data("AID", self.agent_id)
        self.save_data("Energy", self.energy)
        self.hunt()
        self.die()

        self.energy -= 1
        if self.energy < 1:
            self.kill()

    def change_position(self):
        self.there_is_no_escape()
        self.move.rotate_ip(rnd.uniform(-10,10))
        self.pos += self.move.normalize() * self.config.predator_movement_speed * self.config.dT

        

    def hunt(self):
        all_prey = self.in_proximity_accuracy().filter_kind(Prey).filter(lambda x: x[1] < self.config.eating_distance)

        for prey, _ in all_prey:
            prey.kill()
            self.getbabies()
            self.replenish_energy()
    
    def replenish_energy(self):
        self.energy = self.config.default_energy_level

    def getbabies(self):
        if self.config.Predator_Reproduction_Rate > rnd.uniform():
            self.reproduce()

    def die(self):
        if self.config.death_Rate > rnd.uniform():
            self.kill()






class Prey(Agent):
    config: EcosystemConfig

    def on_spawn(self):
        self.move = Vector2(rnd.uniform(),rnd.uniform())
        self.agent_id = "Prey"
    
    def update(self):
        self.save_data("AID", self.agent_id)
        self.save_data("Energy", "N/A")
        self.getbabies()


    def change_position(self):
        self.there_is_no_escape()
        self.move.rotate_ip(rnd.uniform(-10,10))
        self.pos += self.move.normalize() * self.config.prey_movement_speed * self.config.dT

    def getbabies(self):
        if self.config.Prey_Reproduction_Rate > rnd.uniform():
            self.reproduce()



class PreyPredatorSimulation(Simulation):
    pass   

df_raw = (
    PreyPredatorSimulation(
        EcosystemConfig(
            image_rotation = False,
            radius = 30,
            seed = 1,
            duration = SECONDS * 60
        )
    )
    .batch_spawn_agents(100, Prey, images=["Assignment 2/Images/bunny.png"])
    .batch_spawn_agents(10, Predator, images=["Assignment 2/Images/fox.png"])
    .run()
    .snapshots
    .groupby("frame", 'AID')
    .agg(pl.count("AID").alias("AgentCount"))
    .sort("frame", 'AID')
)

print(df_raw)

df_raw.write_csv("Assignment 2/Results/PreyPredator_Chase.csv")
plot = sns.relplot(x = df_raw["frame"], y = df_raw["AgentCount"], hue = df_raw["AID"], kind = "line")
plot.savefig("Assignment 2/Results/PreyPredator_Chase.png", dpi = 300)


