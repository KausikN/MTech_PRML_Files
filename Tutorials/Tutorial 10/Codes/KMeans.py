'''
KMeans
'''

# Imports
import numpy as np

# Main Functions
def dist(pt, c):
    '''
    Calculate the distance of a point to a cluster
    '''
    pt = np.array(pt)
    c = np.array(c)
    return np.sqrt(np.sum((pt - c)**2))

# Driver Code
# Q3
c1_pts = [[4, 2], [3, 2], [2, 3], [-1, 1]]
c2_pts = [[2, 2], [-2, 0], [-1, -1]]

c1 = np.mean(c1_pts, axis=0)
c2 = np.mean(c2_pts, axis=0)

print("C1:", c1, c1_pts)
print("C2:", c2, c2_pts)

it = 2
while(it > 0):
    pts = c1_pts + c2_pts
    c1_pts_old = c1_pts
    c2_pts_old = c2_pts
    c1_pts = []
    c2_pts = []
    print(it)
    for pt in pts:
        d1 = dist(pt, c1)
        d2 = dist(pt, c2)
        print(pt, d1, d2, "C1" if d1 < d2 else "C2")
        if d1 < d2:
            c1_pts.append(pt)
        else:
            c2_pts.append(pt)

    c1 = np.mean(c1_pts, axis=0)
    c2 = np.mean(c2_pts, axis=0)

    print("C1:", c1, c1_pts)
    print("C2:", c2, c2_pts)

    it -= 1
    print()


# Q4 (i)
pts = [
    [-2, -2], [-1, -2], [2, 1], [1, 2]
]
for i in range(len(pts)):
    avgdist = 0.0
    for j in range(len(pts)):
        if i != j:
            # print(pts[i], pts[j], dist(pts[i], pts[j]))
            avgdist += dist(pts[i], pts[j])
    avgdist /= len(pts)-1
    print("Avg dist:", pts[i], avgdist)

# Q4 (ii)
pts = [
    [1, 2], [-2, -2], [1, 0.5]
]
for i in range(len(pts)):
    avgdist = 0.0
    for j in range(len(pts)):
        if i != j:
            print(pts[i], pts[j], dist(pts[i], pts[j]))