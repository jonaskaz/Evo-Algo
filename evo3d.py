import numpy as np
import random
import matplotlib.pyplot as plt


class Population:
    def __init__(self, pop_size, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.pop = self.create_pop(pop_size)
        self.scores = np.zeros(shape=(pop_size))
        self.ax = plt.axes(projection ='3d')
    
    def create_pop(self, pop_size):
        x = np.array([np.random.uniform(self.xmin, self.xmax, 1) for _ in range(pop_size)])
        y = np.array([np.random.uniform(self.ymin, self.ymax, 1) for _ in range(pop_size)])
        return np.append(x, y, axis=1)

    @property
    def pop_size(self):
        return len(self.pop)

    @property
    def x(self):
        return self.pop[:,0]

    @property
    def y(self):
        return self.pop[:,1]

    @staticmethod
    def fitness(x, y):
        return x*(x-3)*(x+2)*np.sin(x) + y*(y-2)*np.sin(y)
    
    def score(self):
        self.scores = np.zeros(shape=(self.pop_size))
        for i, ind in enumerate(self.pop):
            self.scores[i] = self.fitness(ind[0], ind[1])
    
    def select_from_batch(self, batch_size = 10):
        select_coords = None
        select_score = -10000
        for i in np.random.randint(0, self.pop_size, batch_size):
            if self.scores[i] > select_score:
                select_score = self.scores[i]
                select_coords = self.pop[i]
        assert select_coords is not None
        return select_coords
    
    def selection(self, num_to_select):
        self.pop = np.array([self.select_from_batch() for i in range(num_to_select)])

    def reproduce(self, p_cross = 0.8):
        babies = []
        for i in range(0, self.pop_size-1, 2):
            if random.uniform(0, 1) < p_cross:
                babies.append(self.mutate(self.create_baby(self.pop[i], self.pop[i+1])))
        if len(babies) > 0:
            self.pop = np.concatenate([self.pop, babies])
        
    @staticmethod
    def create_baby(ind1, ind2):
        x1, y1 = ind1.tolist()
        x2, y2 = ind2.tolist()
        return [(x1+x2)/2, (y1+y2)/2]
    
    @staticmethod
    def constrain(val, min_val, max_val):
        return min(max_val, max(min_val, val))
    
    def in_range(self, ind):
        return ind[0] >= self.xmin and ind[0] <= self.xmax and ind[1] >= self.ymin and ind[1] <= self.ymax
    
    def constrain_pop(self):
        new_pop = []
        for ind in self.pop:
            if self.in_range(ind):
                new_pop.append(ind)
        self.pop = np.array(new_pop)

    def mutate(self, baby, p_mut = 0.5):
        if random.uniform(0, 1) < p_mut:
            baby[0] = baby[0]*random.uniform(-2, 2)#self.constrain(baby[0]*random.uniform(-1, 2), self.xmin, self.xmax)
            baby[1] = baby[1]*random.uniform(-2, 2)#self.constrain(baby[1]*random.uniform(-1, 2), self.ymin, self.ymax)
        return baby
    
    def plot_optimal(self, xmin=0, xmax=0, ymin=0, ymax=0):
        x = np.arange(xmin, xmax, 0.2)
        y = np.arange(ymin, ymax, 0.2)
        x, y = np.meshgrid(x, y)
        z = self.fitness(x, y)
        self.ax.plot_wireframe(x, y, z)

    def plot_pop(self):
        self.ax.scatter(self.x, self.y, self.scores, marker="x", c="red")
        self.ax.set_title("Population after 6 Epoch")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Fitness")



if __name__ == "__main__":
    #xy_range = [-3, 4, -2, 4] 
    # Use new constrained range
    xy_range = [-1, 4, -2, 2]
    pop = Population(1000, * xy_range)
    pop.score()
    epochs = 100
    for i in range(epochs):
        pop.selection(int(pop.pop_size/1.28))
        print(pop.pop_size)
        pop.reproduce()
        print(pop.pop_size)
        pop.score()
        pop.constrain_pop()
    pop.selection(int(pop.pop_size/1.3))
    pop.score()
    pop.constrain_pop()
    pop.plot_pop()
    pop.plot_optimal(-3, 4, -2, 4)
    plt.show()
