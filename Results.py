import matplotlib.pyplot as plt
import numpy as np


# macbook = (2200 / 24, 800 / 11, 1800 / 26)
# pi = (2400 / 24, 700 / 11, 1700 / 26)

macbook = (1600 / 18, 700 / 11, 400 / 9)
pi = (1600 / 18, 700 / 11, 400 / 9)

ind = np.arange(len(macbook))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width / 2, macbook, width,
                label='Macbook Pro')
rects2 = ax.bar(ind + width / 2, pi, width,
                label='Raspberry Pi')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title('Detection Accuracy (Night)')
ax.set_xticks(ind)
ax.set_xticklabels(('Tilton', 'Saluda', 'Derry'))
ax.legend()




fig.tight_layout()

plt.show()