import matplotlib.pyplot as plt
import numpy as np


# macbook_day = (2200 / 24, 800 / 11, 1800 / 26)
# pi_day = (2400 / 24, 700 / 11, 1700 / 26)
#
# macbook_night = (1600 / 18, 700 / 11, 400 / 9)
# pi_night = (1600 / 18, 700 / 11, 400 / 9)
#
# ind = np.arange(len(macbook_day))  # the x locations for the groups
# width = 0.35  # the width of the bars
#
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
# rects1 = ax1.bar(ind - width / 2, macbook_day, width,
#                 label='Macbook Pro')
# rects2 = ax1.bar(ind + width / 2, pi_day, width,
#                 label='Raspberry Pi')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax1.set_ylabel('Accuracy (%)')
# ax1.set_title('Vehicle Detection Accuracy (Day)')
# ax1.set_xticks(ind)
# ax1.set_xticklabels(('Tilton', 'Saluda', 'Derry'))
# ax1.legend()
#
# rects1 = ax2.bar(ind - width / 2, macbook_night, width,
#                 label='Macbook Pro')
# rects2 = ax2.bar(ind + width / 2, pi_night, width,
#                 label='Raspberry Pi')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax2.set_ylabel('Accuracy (%)')
# ax2.set_title('Vehicle Detection Accuracy (Night)')
# ax2.set_xticks(ind)
# ax2.set_xticklabels(('Tilton', 'Saluda', 'Derry'))
# ax2.legend()
#
#
#
# fig.tight_layout()
#
# plt.show()

mac = open('mac_results.txt', 'r')
lines = mac.readlines()
mac_speeds = []
for line in lines:
    mac_speeds.append(float(line.strip()))

pi = open('pi_results.txt', 'r')
lines = pi.readlines()
pi_speeds = []
for line in lines:
    pi_speeds.append(float(line.strip()))
print(mac_speeds)
print(len(mac_speeds))
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.hist(mac_speeds, bins =np.linspace(0, 100, 21))
ax1.plot(100 * [45], np.linspace(0, 7, 100), '-r', label="Speed Limit")
ax2.hist(pi_speeds, bins =np.linspace(0, 100, 21))
ax2.plot(100 * [45], np.linspace(0, 14, 100), '-r', label="Speed Limit")
ax1.set_xlabel("Speed (MPH)")
ax1.set_ylabel("Total occurrences")
ax1.set_title("Speed Distribution (Macbook Pro)")
ax1.legend()
ax2.set_xlabel("Speed (MPH)")
ax2.set_ylabel("Total occurrences")
ax2.set_title("Speed Distribution (Raspberry Pi)")
ax2.legend()
fig.tight_layout()
plt.show()
