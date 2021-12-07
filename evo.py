import numpy as np
import matplotlib.pyplot as plt
import random


class Population:
    def __init__(self):
        self.pop = np.random.randint(0, 1000, 1000)


def eval_fitness(pop, max_kills=300):
    to_delete = []
    for i, val in enumerate(pop):
        if i >= max_kills:
            break
        if val < 699 or val > 700:
            to_delete.append(i)
    pop = np.delete(pop, to_delete)
    return pop, len(to_delete)


def pair(pop):
    pairs = np.array_split(pop, 2)
    for p1, p2 in zip(pairs[0], pairs[1]):
        yield p1, p2


def reproduce(pop, num_to_create):
    pair_iter = pair(pop)
    children = []
    for _ in range(num_to_create):
        children.append(np.mean(next(pair_iter)))
    return mutate(np.append(pop, children))


def mutate(pop, num_to_mutate=40):
    idx = np.random.randint(0, len(pop), num_to_mutate)
    for i in idx:
        pop[i] += random.randint(-100, 100)
    return pop


def evolve(pop):
    pop, num_killed = eval_fitness(pop)
    return reproduce(pop, num_killed)


if __name__ == "__main__":
    num_epochs = 500
    population = Population()
    print(np.mean(population.pop))
    pop_means = []
    for _ in range(num_epochs):
        pop = evolve(population.pop)
        print(np.mean(pop))
        pop_means.append(np.mean(pop))
        population.pop = pop
    plt.plot(range(num_epochs), pop_means)
    plt.xlabel("Generation")
    plt.ylabel("Mean of population")
    plt.title("Evolutionary Algorithm With Fitness defined as distance from 700")
    plt.show()
