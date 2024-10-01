from enum import Enum, auto
import random
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

EPS = sys.float_info.epsilon
SPEEDFACTOR = 1
SECONDS = 600
# rnd.seed(5)

class EcosystemConfig(Config):
    # Rabbit Config
    prey_movement_speed = 10
    Prey_Reproduction_Rate = 0.0032
    prey_energy_level = 5.5 * 60
    prey_hunger_threshold = 0.60 * prey_energy_level   # Threshold for when prey starts looking for grass (hungry)
    
    # Fox Config
    predator_movement_speed = 25
    stochastic_death_Rate = 0.0001
    default_energy_level = 3.855 * 60
    hunger_threshold = 0.65 * default_energy_level   # Threshold for when the predator starts to chase (hungry)
    Predator_Reproduction_Rate = 0.03345
    eating_distance = 16
    life_expectancy = 4000

    # General Config
    dT = 0.1
    
    # Grass config
    grass_replenish_time = 400


class Predator(Agent):
    config: EcosystemConfig

    def on_spawn(self):
        self.move = Vector2(rnd.uniform(-1,1),rnd.uniform(-1,1))
        self.agent_id = "Predator"
        self.energy = self.config.default_energy_level
        self.life_expectancy = self.config.life_expectancy + rnd.uniform(-500, 500)

    def update(self):
        self.save_data("AID", self.agent_id)
        self.save_data("Energy", self.energy)
        if self.energy < self.config.hunger_threshold:
            self.chase()   
        self.hunt()
        self.die()

        self.energy -= 1
        if self.energy < 1:
            self.kill()

    def change_position(self):
        self.there_is_no_escape()
        self.move.rotate_ip(rnd.uniform(-10,10))
        self.movement_speed = -1 / (self.config.default_energy_level ** 2 / self.config.predator_movement_speed) * (self.config.default_energy_level - self.energy) ** 2 + self.config.predator_movement_speed
        self.pos += self.move.normalize() * self.movement_speed * self.config.dT

    
    def chase(self):
        prey = self.in_proximity_accuracy().filter_kind(Prey)
        nearby_prey = list(prey)

        if len(nearby_prey) == 0:
            return False
        
        nearest_prey, _ = min(nearby_prey, key=lambda x: x[1])
        self.move = nearest_prey.pos - self.pos

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
        if self.config.stochastic_death_Rate > rnd.uniform() or self.life_expectancy < 1:
            self.kill()
        else:
            self.life_expectancy -= 1


class Grass(Agent):
    config: EcosystemConfig
    _iseaten = False

    def eatable(self):
        return not self._iseaten

    def on_spawn(self):
        self.agent_id = "Grass"
        self.replenish_timer = self.config.grass_replenish_time + rnd.uniform(-100, 100)
        self.change_image(0)
    
    def update(self):
        self.save_data("AID", self.agent_id)
        self.save_data("Energy", -1)

    def eat(self):
        self._iseaten = True
        self.change_image(1)
    
    def change_position(self):
        if self.eatable():
            return
        
        if self.replenish_timer > 0:
            self.replenish_timer -= 1
            return
        
        self._iseaten = False
        self.replenish_timer = self.config.grass_replenish_time + rnd.uniform(-100, 100)

        self.change_image(0)


class Prey(Agent):
    config: EcosystemConfig

    def on_spawn(self):
        self.move = Vector2(rnd.uniform(),rnd.uniform())
        self.energy = self.config.prey_energy_level + rnd.uniform(-50, 200)
        self.agent_id = "Prey"
    
    def update(self):
        self.getbabies()
        self.save_data("AID", self.agent_id)
        self.save_data("Energy", self.energy)

    def replenish_energy(self):
        self.energy = self.config.prey_energy_level

    def find_grass(self):
        if self.config.prey_hunger_threshold < self.energy:
            return False
                
        grass_in_proximity = self.in_proximity_accuracy().filter_kind(Grass).filter(lambda x: x[0].eatable()).collect_set()
        if len(grass_in_proximity) == 0:
            return False
        
        nearest_grass, dist = min(grass_in_proximity, key=lambda x: x[1])
        if dist < 8:
            nearest_grass.eat()
            self.replenish_energy()
            return False

        self.move = nearest_grass.pos - self.pos
        self.pos += self.move.normalize()
        
        return True
    

    def change_position(self):
        self.there_is_no_escape()
        self.energy -= 1

        if self.energy < 1:
            self.kill()

        prey_moved = self.find_grass()
        self.movement_speed = -1 / (self.config.prey_energy_level ** 2 / self.config.prey_movement_speed) * (self.config.prey_energy_level - self.energy) ** 2 + self.config.prey_movement_speed
        if prey_moved:
            return

        self.move.rotate_ip(rnd.uniform(-10,10))
        self.pos += self.move.normalize() * self.movement_speed * self.config.dT

    def getbabies(self):
        if self.config.Prey_Reproduction_Rate > rnd.uniform():
            self.reproduce()

# functions not used currently, might be useful for comparison later
class PreyPredatorSimulation(Simulation):    
    # generate the grass in a circle
    def generate_circle_coordinates(self, center_x, center_y, radius, n_objects):
        coordinates = []
        angle_increment = 2 * math.pi / n_objects

        for i in range(n_objects):
            angle = i * angle_increment
            x = int(center_x + radius * math.cos(angle) - 8)  # Adjust for sprite size
            y = int(center_y + radius * math.sin(angle) - 8)  # Adjust for sprite size
            coordinates.append((x, y))

        return coordinates

    # Generate a patch of random grass within a circle
    def generate_patch_coordinates(self, center, radius, n_objects):
        coordinates = []
        
        cx,cy = center
        for i in range(n_objects):
            r = radius * math.sqrt(random.uniform(0,1))
            theta = random.uniform(0,2 * math.pi)
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            
            coordinates.append((x,y))
        
        return coordinates
        

    def spawn_grass(self, count):
        """         coordinates = self.generate_circle_coordinates(375, 375, 200, count) """
        
        coordinates = self.generate_patch_coordinates( (375,375), 200, count )

        all_images = self._load_images(["Assignment 2/Images/grass.png", "Assignment 2/Images/grass_eaten.png"])
        for i in range(count):
            agent = Grass(images=all_images, simulation=self)
            agent.pos = Vector2(coordinates[i][0], coordinates[i][1])


        return self

for i in range(20):
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
        .batch_spawn_agents(10, Predator, images=["Assignment 2/Images/fox.png", "Assignment 2/Images/fox_chasing.png"])
        .batch_spawn_agents(80, Grass, images = ["Assignment 2/Images/grass.png", "Assignment 2/Images/grass_eaten.png"])
        .run()
        .snapshots
        .groupby("frame", 'AID')
        .agg(pl.count("AID").alias("AgentCount"))
        .sort("frame", 'AID') 
    )

    df_raw.write_csv(f"Assignment 2/20Runs/PreyPredator_Grass_Rand{i}.csv")
    plot = sns.relplot(x = df_raw["frame"], y = df_raw["AgentCount"], hue = df_raw["AID"], kind = "line")
    plot.savefig(f"Assignment 2/20Runs/PreyPredator_Grass_Rand{i}.png", dpi = 300)

for i in range(20):
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
        .batch_spawn_agents(10, Predator, images=["Assignment 2/Images/fox.png", "Assignment 2/Images/fox_chasing.png"])
        .spawn_grass(80)
        #.batch_spawn_agents(80, Grass, images = ["Assignment 2/Images/grass.png", "Assignment 2/Images/grass_eaten.png"])
        .run()
        .snapshots
        .groupby("frame", 'AID')
        .agg(pl.count("AID").alias("AgentCount"))
        .sort("frame", 'AID') 
    )

    df_raw.write_csv(f"Assignment 2/20Runs/PreyPredator_Grass{i}.csv")
    plot = sns.relplot(x = df_raw["frame"], y = df_raw["AgentCount"], hue = df_raw["AID"], kind = "line")
    plot.savefig(f"Assignment 2/20Runs/Results/PreyPredator_Grass{i}.png", dpi = 300)

