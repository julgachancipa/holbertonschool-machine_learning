#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

color_list = ['red', 'yellow', '#ff8000', '#ffe5b4']
names = ['Farrah', 'Fred', 'Felicia']
f_names = ['apples', 'bananas', 'oranges', 'peaches']

for i in range(len(fruit)):
    plt.bar(names, fruit[i],
            bottom=np.sum(fruit[:i], axis=0),
            color=color_list[i % len(color_list)],
            label=f_names[i],
            width=0.5)
plt.yticks(np.arange(0, 81, step=10))
plt.ylim(0, 80)
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.legend(loc="upper right")
plt.show()
