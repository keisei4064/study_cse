import math
import matplotlib.pyplot

xs = [0.1 * i - 5 for i in range(100)]
ys = [math.sin(x) for x in xs]

matplotlib.pyplot.plot(xs, ys)
matplotlib.pyplot.show()
