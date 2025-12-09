import xml.etree.ElementTree as ET
from svgoutline import svg_to_outlines
import json
import numpy as np

def get_only_points(file):
    tree = ET.parse(file)
    root = tree.getroot()
    h = float(root.attrib['height'].replace("mm","").replace("px",""))
    w = float(root.attrib['width'].replace("mm","").replace("px",""))
    print(h, w)
    outlines = svg_to_outlines(root ,pixels_per_mm=1)
    print(len(outlines))
    return outlines, h, w

def get_points(file):

    outlines, h, w = get_only_points(file)

    print(outlines)

    all_points = []
    all_normals = []

    for i in range(len(outlines)):
        all_points += outlines[i][2]

    seen = set()
    unique_points = []
    for p in all_points:
        pt = tuple(p)
        if pt not in seen:
            seen.add(pt)
            unique_points.append(p)

    all_points = unique_points

    for i in range(len(all_points)):
        all_points[i] = [2 * (all_points[i][0] / w) - 1.0, 2 * (all_points[i][1] / w) - (h / w)]

    all_points = np.array(all_points)

    filter = False
    if filter:
        print(len(all_points))
        filtered = []
        for pi in all_points:
            success = True
            for pj in all_points:
                if(pi[0] == pj[0] and pi[1] == pj[1]):
                    success = False
                    continue
            if success:
                filtered.append(pi)

    interpolate = False
    if interpolate:
        interpsteps = 51
        print(len(all_points))
        new_points = []
        for i in range(len(all_points) - 1):
            new_points.append(all_points[i])
            for j in range(interpsteps):
                new_points.append(all_points[i] + (all_points[i + 1] - all_points[i]) * (1.0/interpsteps) * j)
        all_points = np.array(new_points)
        print(len(all_points))


    clusterstrip = False
    if clusterstrip:
        print(len(all_points))
        stripped = []
        for pi in all_points:
            success = True
            for pj in all_points:
                if(not (pi[0] == pj[0] and pi[1] == pj[1]) and np.linalg.norm(pi - pj) < 0.0004):
                    success = False
                    continue
            if success:
                stripped.append(pi)

        all_points = np.array(stripped)
        print(len(all_points))
        #all_points = all_points[np.r_[True, >=0.3]]

    for i in range(len(all_points)):
        if i < len(all_points) - 1:
            dx = all_points[i + 1][0] - all_points[i][0]
            dy = all_points[i + 1][1] - all_points[i][1]
            normal = (dy, -dx)
            all_normals.append(normal / np.linalg.norm(normal))
    all_normals.append((0, 0))
    #print(all_normals)
    return all_points, np.array(all_normals)

if __name__ == "__main__":
    points, normals = get_points("svgs/Mushroom-14.svg")



