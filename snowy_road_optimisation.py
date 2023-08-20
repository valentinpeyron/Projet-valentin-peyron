import heapq
from collections import defaultdict, deque


def dijkstra_neige(n, edges, start, end):#
    distances = [float("inf") for _ in range(n)]
    distances[start] = 0
    q = deque()
    q.append(start)

    while len(q):
        node = q.popleft()
        for b, w, c in edges[node]:

            dist= max(c, distances[node])
            if dist < distances[b]:
                distances[b] = dist
                q.append(b)

    if distances[end] == float("inf"):
        print("Impossible")
        exit(0)
    else:
        # filtered_edge = [t for t in edges if t[3] <= distances[end]]
        return distances[end]


def dijkstra_longueur(n, edges,depth, start, end):

    distances = [float("inf") for _ in range(n)]
    distances[start] = 0
    heap = []
    heapq.heappush(heap, start)

    while heap:
        
        node = heapq.heappop(heap)
        for b, w, c in edges[node]:
            dist= w + distances[node]
            if dist < distances[b] and c <= depth:
                distances[b] = dist
                heapq.heappush(heap, b)
    return distances[end]

n,m,s,t = [int(i) for i in input().split()]
s = s - 1
t = t - 1
edges_neige = defaultdict(list)

for i in range(m):
    a, b, c, k = list(map(int,input().split()))
    edges_neige[a-1].append((b-1,c,k))
    edges_neige[b-1].append((a-1,c,k))
#edges_2 = edges_neige[:]#
depth  = dijkstra_neige(n,edges_neige, s, t)
if depth =="Impossible":
        print(depth)
else:
        length = dijkstra_longueur(n,edges_neige, depth, s, t)
        print(depth,length)
