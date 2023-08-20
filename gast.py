from collections import defaultdict
import sys

sys.setrecursionlimit(2500)
n,m,initial_fuel = [int(i) for i in input().split()]

def rec(arr, dp, i,j):
    if i < 0 :
        return -1
    if (i,j) in dp:
        return dp[(i,j)]
    if j == n - 1 : 
        dp[(i,j)]  = 0
        return dp[(i,j)] 
    dp[(i,j)]  = rec(arr,dp,i-1,j+1)
    
    for end,can,fuel in arr[j] :
        index = min(i- (end-j) + fuel, n - (j + 1))
        ans = rec(arr,dp,index,end)
        if ans != -1:
            dp[(i,j)] = max(dp[(i,j)],ans + can)
    return dp[(i,j)]
            
            


hitch = defaultdict(list)

for i in range(m):
    a, b, c, f = list(map(int,input().split()))
    hitch[a-1].append((b-1,c,f))

dp = {}
answer = rec(hitch,dp,min(initial_fuel,n-1),0)

if answer == -1 : 
    print("Impossible")
else : 
    print(answer)