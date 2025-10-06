"""
Given two vectors:
x = [2, 3, 5, 7,5,6,3,1,19,11,12,16,15,5,2,1,7,2,4,5,6,12,11,2,3]
 y = [1, 1, 2, 3, 2, 3, 5, 7,5,6,3,1,19,11,10,16,14,5,2,1,8,2,4,5,2]
1. Write a function minkowski_distance(x, y, p) that calculates the Minkowski distance of order p.
2. Using this function, calculate the following distances between  x and y :
L1 distance (p=1)
L2 distance (p=2)
L3 distance (p=3)
3. Print the results with appropriate labels.
"""

def minkowski_distance(p):
    x = [2, 3, 5, 7,5,6,3,1,19,11,12,16,15,5,2,1,7,2,4,5,6,12,11,2,3]
    y = [1, 1, 2, 3, 2, 3, 5, 7,5,6,3,1,19,11,10,16,14,5,2,1,8,2,4,5,2]

    dist = []
    for i,j in zip(x, y):
        dist.append(abs(i-j))
    # print(dist)
    dist_sq = []
    for i in dist:
        dist_sq.append(i**p)

    dist_sum = 0
    for i in dist_sq:
        dist_sum += i

    dist_ans = dist_sum**(1/p)

    if p == 1:
        print(f"manhatten distance is  {dist_ans} ")
    elif p == 2:
        print(f"eucledian distance is {dist_ans}")
    else:
        print(f"distance for p = {p} is {dist_ans}")

start = (int)(input("enter starting point: "))
end = (int)(input("enter ending point: "))
for i in range(start,end+1):
    minkowski_distance(i)
