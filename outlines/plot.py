import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from outline import get_points, get_only_points
import xml.etree.ElementTree as ET
import json
import numpy as np

plt.xlabel('Width')
plt.ylabel('Height')

'''
individual = False
if individual:
    outlines, h, w = get_only_points("svgs/kittyclean.svg")
    for i in range(len(outlines)):
        points = np.array(outlines[i][2])
        for i in range(len(points)):
            points[i] = [2 * (points[i][0] / w) - 1.0, 2 * (points[i][1] / w) - (h / w)]
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(points[:,0], -points[:,1])
        plt.grid()
        plt.show()
'''

points, normals = get_points("svgs/kittyclean.svg")
points = np.roll(points, 500)
normals = np.roll(normals, 500)
print(len(points))
fig = plt.figure()
ax = fig.gca()

#points[:,1] = -points[:,1]
ax.scatter(points[:,0], points[:,1])
plt.grid()
ax.plot(*np.c_[points, points + 0.1 * normals, points*np.nan].reshape(-1, 2).T, 'k')
#ax.plot(*np.c_[points, 1.1 * points, points*np.nan].reshape(-1, 2).T, 'k')
plt.show()
structure = {
    "points": points.tolist(),
    "normals" : normals.tolist()
}
print(len(points))
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(structure, f, ensure_ascii=False, indent=4)