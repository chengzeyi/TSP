# -*- coding: utf-8 -*-
import numpy

# 城市类
class City(object):
    def __init__(self, x, y, name="Unknown City Name"):
        self.x = x
        self.y = y
        self.name = name


# 世界类
class World(object):
    def __init__(self):
        self.cities = []

    @property
    def city_num(self):
        return len(self.cities)

    def reset_distance_graph(self):
        self.distance_graph = numpy.zeros((self.city_num, self.city_num))
        for i in range(self.city_num):
            for j in range(self.city_num):
                temp_distance = pow(self.cities[i].x - self.cities[j].x, 2) + pow(self.cities[i].y - self.cities[j].y, 2)
                temp_distance = pow(temp_distance, 0.5)
                self.distance_graph[i][j] = temp_distance

    def reset_pheromone_graph(self):
        self.pheromone_graph = numpy.ones((self.city_num, self.city_num))
