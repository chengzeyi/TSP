# -*- coding: utf-8 -*-
import random

import math
import numpy
import sys

# 仅用于继承
class TSPSolver(object):
    def __init__(self, name, active_key):
        self.name = name
        self.active_key = active_key

    def solve(self, world):
        pass


# 蚂蚁类
class Ant(object):
    def __init__(self, id, world, alpha=1.0, beta=2.0):
        self.id = id
        self.world = world
        self.alpha = alpha
        self.beta = beta
        self.reset_status()

    # 重置状态
    def reset_status(self):
        self.path = []
        self.total_distance = 0.0
        self.current_city = random.randint(0, self.world.city_num - 1)
        self.path.append(self.current_city)
        self.open_city_table = [True for i in range(self.world.city_num)]
        self.open_city_table[self.current_city] = False
        self.move_count = 1
        self.world.pheromone_graph = numpy.ones((self.world.city_num, self.world.city_num))

    # 选择要去的下一个城市
    def choose_next_city(self):
        next_city = -1
        select_city_probs = [0.0 for i in range(self.world.city_num)]
        total_prob = 0.0

        for i in range(self.world.city_num):
            if self.open_city_table[i]:
                try:
                    select_city_probs[i] = pow(self.world.pheromone_graph[self.current_city][i], self.alpha) * \
                        pow(1.0 / self.world.distance_graph[self.current_city][i], self.beta)
                    total_prob += select_city_probs[i]
                except ZeroDivisionError as e:
                    print(e)
                    print("ant ID: {}, current city: {}, target city: {}".format(self.id, self.current_city, i))
                    sys.exit(1)

        # 轮盘选择城市
        if (total_prob > 0.0):
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(self.world.city_num):
                if (self.open_city_table[i]):
                    temp_prob -= select_city_probs[i]
                    if (temp_prob < 0.0):
                        next_city = i
                        break

        if next_city == -1:
            for i in range(self.world.city_num):
                if self.open_city_table[i]:
                    next_city = i
                    break

        return next_city

    # 计算路径距离
    def cal_distance(self):
        distance = 0.0
        for i in range(len(self.path)):
            start, end = self.path[i], self.path[i - 1]
            distance += self.world.distance_graph[start][end]
        return distance

    # 移动操作
    def move(self, next_city):
        self.path.append(next_city)
        self.open_city_table[next_city] = False
        self.total_distance += self.world.distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1

    def search_path(self):
        self.reset_status()

        while self.move_count < self.world.city_num:
            next_city = self.choose_next_city()
            self.move(next_city)

        self.total_distance += self.cal_distance()


# 蚁群算法求解器
class AntSolver(TSPSolver):
    def __init__(self, ant_num=50, max_iter=50, alpha=1.0, beta=2.0, rho=0.5, q=100.0):
        super().__init__("Ant Algo", "a")
        self.ant_num = ant_num
        self.max_iter = max_iter
        self.alpha=alpha
        self.beta=beta
        self.rho=rho
        self.q=q

    def solve(self, world):
        world.reset_pheromone_graph()
        ants = [Ant(id, world, self.alpha, self.beta) for id in range(self.ant_num)]
        best_ant_distance = float("inf")
        best_ant_path = []
        for i in range(self.max_iter):
            for ant in ants:
                ant.search_path()
                if ant.total_distance < best_ant_distance:
                    best_ant_distance = ant.total_distance
                    best_ant_path = ant.path
            self.update_pheromone_graph(ants, world)
            print("[Ant Algo]iter order: {}, total distance of best path: {}".format(i, best_ant_distance))
        return best_ant_path

    def update_pheromone_graph(self, ants, world):
        temp_pheromone = numpy.zeros((world.city_num, world.city_num))
        for ant in ants:
            for i in range(1, world.city_num):
                start, end = ant.path[i - 1], ant.path[i]
                temp_pheromone[start][end] += self.q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]

        for i in range(world.city_num):
            for j in range(world.city_num):
                world.pheromone_graph[i][j] *= self.rho
                world.pheromone_graph[i][j] += temp_pheromone[i][j]


# 计算当前个体的适应值和路径距离
class Fitness(object):
    def __init__(self, world, path):
        self.world = world
        self.path = path
        self.distance = 0.0
        self.fitness = 0.0

    def get_path_distance(self):
        if self.distance == 0.0:
            path_distance = 0.0
            for i in range(len(self.path)):
                from_city = self.path[i]
                to_city = None
                if i + 1 < len(self.path):
                    to_city = self.path[i + 1]
                else:
                    to_city = self.path[0]
                path_distance += self.world.distance_graph[from_city][to_city]
            self.distance = path_distance
        return self.distance

    def get_path_fitness(self):
        if self.fitness == 0.0:
            self.fitness = 1.0 / self.get_path_distance()
        return self.fitness


# 遗传算法TSP求解器
class GeneticSolver(TSPSolver):
    def __init__(self, pop_size=100, elite_size=20, mutation_rate=0.01, max_iter=100):
        super().__init__("Genetic Algo", "g")
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter

    def create_path(self, world):
        path = random.sample(range(world.city_num), world.city_num)
        return path

    def init_population(self, pop_size, world):
        population = []
        for i in range(pop_size):
            population.append(self.create_path(world))
        return population

    def rank_paths(self, world, population):
        fitness_results = []
        for i in range(len(population)):
            fitness_results.append((i, Fitness(world, population[i]).get_path_fitness()))
        fitness_results.sort(key=lambda fitness_result: fitness_result[1], reverse=True)
        return fitness_results

    def select(self, pop_ranked, elite_size):
        selection_results = []
        fitness_sum = sum((fitness for (_, fitness) in pop_ranked))
        cum_sum = numpy.cumsum([fitness for (_, fitness) in pop_ranked])
        cum_rate = cum_sum / fitness_sum

        for i in range(elite_size):
            selection_results.append(pop_ranked[i][0])
        for _ in range(len(pop_ranked) - elite_size):
            pick = random.random()
            for i in range(len(pop_ranked)):
                if pick <= cum_rate[i]:
                    selection_results.append(pop_ranked[i][0])
                    break
        return selection_results

    def get_mating_pool(self, population, selection_results):
        mating_pool = [population[selection_result] for selection_result in selection_results]
        return mating_pool

    def breed(self, parent1, parent2):
        gene_a = int(random.random() * len(parent1))
        gene_b = int(random.random() * len(parent1))
        start_gene = min(gene_a, gene_b)
        end_gene = max(gene_a, gene_b)
        child_p1 = [parent1[i] for i in range(start_gene, end_gene)]
        child_p2 = [item for item in parent2 if item not in child_p1]
        return child_p1 + child_p2

    def breed_population(self, mating_pool, elite_size):
        children = [mating_pool[i] for i in range(elite_size)]
        pool = random.sample(mating_pool, len(mating_pool))
        children.extend((self.breed(pool[i], pool[-1 - i]) for i in range(len(mating_pool) - elite_size)))
        return children

    def mutate(self, individual, mutation_rate):
        for swapped in range(len(individual)):
            if random.random() < mutation_rate:
                swap_with = int(random.random() * len(individual))
                individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
        return individual

    def mutate_population(self, population, mutation_rate):
        mutated_pop = [self.mutate(ind, mutation_rate) for ind in population]
        return mutated_pop

    def get_next_generation(self, world, current_generation, elite_size, mutation_rate):
        pop_ranked = self.rank_paths(world, current_generation)
        selection_results = self.select(pop_ranked, elite_size)
        mating_pool = self.get_mating_pool(current_generation, selection_results)
        children = self.breed_population(mating_pool, elite_size)
        next_generation = self.mutate_population(children, mutation_rate)
        return next_generation

    def solve(self, world):
        pop = self.init_population(self.pop_size, world)
        for i in range(self.max_iter):
            pop = self.get_next_generation(world, pop, self.elite_size, self.mutation_rate)
            print("[Genetic Algo]iter order: {}, total distance of best path: {}".format(i, 1 / self.rank_paths(world, pop)[0][1]))
        best_path = pop[self.rank_paths(world, pop)[0][0]]
        return best_path


class SimAnnealSolver(TSPSolver):
    def __init__(self, temperature=10000.0, cooling_rate=0.999, abs_temperature=0.00001):
        super().__init__("Sim Anneal Algo", "s")
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.abs_temperature = abs_temperature

    def get_total_distance(self, world, path):
        total_distance = 0
        for i in range(len(path)):
            total_distance += world.distance_graph[path[i - 1]][path[i]]
        return total_distance

    def solve(self, world):
        path = [i for i in range(world.city_num)]
        total_distance = self.get_total_distance(world, path)
        temperature = self.temperature
        iter = 0
        while temperature > self.abs_temperature:
            next_path = self.get_next_path(path)
            delta_distance = self.get_total_distance(world, next_path) - total_distance
            if delta_distance < 0 or (delta_distance > 0 and math.exp(-delta_distance / temperature) >
                                      random.random()):
                path = next_path
                total_distance += delta_distance
            temperature *= self.cooling_rate
            print("[Sim Anneal Algo]iter order: {}, total distance of best path: {}".format(iter, total_distance))
            iter += 1
        return path

    def get_next_path(self, path):
        next_path = path.copy()
        first_random_city = random.randint(1, len(path) - 1)
        second_random_city = random.randint(1, len(path) - 1)
        next_path[first_random_city], next_path[second_random_city] = next_path[second_random_city], next_path[first_random_city]
        return next_path
