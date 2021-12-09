import numpy as np
import matplotlib.pyplot as plt
import random


def get_fitness(individual, target=700):
    return abs(individual-target)


def selection(pop, k=3):
    new_pop = []
    for i in range(0, len(pop), k):
        best_score=1e20
        best_individual=0
        for individual in pop[i:i+k]:
            if get_fitness(individual) < best_score:
                best_score = get_fitness(individual)
                best_individual = individual
        new_pop.append(best_individual)
    num_killed = len(pop)-len(pop)//k
    return new_pop, num_killed


def pair(pop):
    random.shuffle(pop)
    pairs = np.array_split(pop, 2)
    for p1, p2 in zip(pairs[0], pairs[1]):
        yield p1, p2


def reproduce(pop, num_to_create):
    pair_iter = pair(pop)
    children = []
    for _ in range(num_to_create):
        try:
            children.append(np.mean(next(pair_iter)))
        except StopIteration:
            pair_iter = pair(pop)
            children.append(np.mean(next(pair_iter)))
    return mutate(np.append(pop, children))


def mutate(pop, p_mut=0.5, min_add=-200, max_add=200):
    for i in range(len(pop)):
        if np.random.random() < p_mut:
            pop[i] += random.randint(min_add, max_add)
    return pop


def evolve(pop, num_gens):
    pop_means = []
    for _ in range(num_gens):
        pop_means.append(np.mean(pop))
        pop, num_killed = selection(pop)
        pop = reproduce(pop, num_killed)
    return pop, pop_means


if __name__ == "__main__":
    pop = np.random.randint(0, 1000, 1000)
    num_gens = 25
    pop, pop_means = evolve(pop, num_gens)
    plt.plot(range(num_gens), pop_means)
    plt.xlabel("Generation")
    plt.ylabel("Mean of population")
    plt.title("Evolutionary algorithm with fitness defined as distance from 700")
    plt.show()
