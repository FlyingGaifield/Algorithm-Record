# 参考
# https://www.kancloud.cn/kancloud/pack/70124
# https://zhuanlan.zhihu.com/p/35643721
# FlyingGaifield
def knapsack_01(N, V, cost, value):
    # 01背包问题
    # 题目：有N件物品和一个容量为V的背包。第i件物品的费用是c[i]，价值是w[i]。求解将哪些物品装入背包可使价值总和最大。
    # 状态转移方程 dp[i][v]=max{dp[i-1][v],dp[i-1][v-c[i]]+w[i]}
    dp = [[0]*(V+1) for i in range(N+1)]
    for i in range(1,N+1):
        for j in range(1,V+1):
            if j >= cost[i-1]:
                dp[i][j]= max(dp[i-1][j], dp[i-1][j-cost[i-1]]+value[i-1] )
            else:
                dp[i][j] = dp[i-1][j]
    #print(dp)
    return dp[-1][-1]

def knapsack_01_1d(N, V, cost, value):
    # 01背包问题
    # 题目：有N件物品和一个容量为V的背包。第i件物品的费用是c[i]，价值是v[i]。求解将哪些物品装入背包可使价值总和最大。
    # 状态转移方程 dp[i][j]=max{dp[i-1][j],dp[i-1][j-c[i]]+v[i]}
    # 在01背包的问题上优化空间
    dp = [0]*(V+1)
    for i in range(1,N+1):
        for j in range(V,-1,-1): # 主要修改在这里（逆序，确保之前的都是i-1# ），根据计算方法，只需要一维数组就行
            if j >= cost[i-1]:
                dp[j]= max(dp[j], dp[j-cost[i-1]]+value[i-1] )
            else:
                dp[j] = dp[j]
    #print(dp)
    return dp[-1]

def knapsack_complete(N, V, cost, value):
    # 完全背包问题
    # 题目 有N种物品和一个容量为V的背包，每种物品都有无限件可用。第i种物品的费用是c[i]，价值是v[i]。求解将哪些物品装入背包可使这些物品的费用总和不超过背包容量，且价值总和最大
    # 状态转移方程 dp[i][j]= max{dp[i-1][j-k*c[i]] + k*v[i] | 0<=k*c[i]<=V }
    # 因为可以拿取无限件，因此可以修改01问题中的转移方程为： dp[i][v]=max{dp[i-1][v],dp[i][v-c[i]]+w[i]}， 为什么不是dp[i-1][v-c[i]]？ 因为这个值可以通过 dp[i][v-c[i]]计算
    dp = [[0]*(V+1) for i in range(N+1)]
    for i in range(1,N+1):
        for j in range(1,V+1):
            if j >= cost[i-1]:
                dp[i][j]= max(dp[i-1][j], dp[i][j-cost[i-1]]+value[i-1] )
            else:
                dp[i][j] = dp[i-1][j]
    #print(dp)
    return dp[-1][-1]

def knapsack_complete_1d(N, V, cost, value):
    # 完全背包问题
    # 题目 有N种物品和一个容量为V的背包，每种物品都有无限件可用。第i种物品的费用是c[i]，价值是v[i]。求解将哪些物品装入背包可使这些物品的费用总和不超过背包容量，且价值总和最大
    # 状态转移方程 dp[i][j]= max{dp[i-1][j-k*c[i]] + k*v[i] | 0<=k*c[i]<=V }
    # 在完全背包的问题上优化空间
    dp = [0]*(V+1)
    for i in range(1,N+1):
        for j in range(1,V+1): # 主要修改在这里(相比于01是正序，因为可以选择自己)，根据计算方法，只需要一维数组就行
            if j >= cost[i-1]:
                dp[j]= max(dp[j], dp[j-cost[i-1]]+value[i-1] )
            else:
                dp[j] = dp[j]
    #print(dp)
    return dp[-1]

def knapsack_multi_1d(N, V, cost, value, num):
    # 多重背包问题
    # 题目 有N种物品和一个容量为V的背包。第i种物品最多有n[i]件可用，每件费用是c[i]，价值是v[i]。求解将哪些物品装入背包可使这些物品的费用总和不超过背包容量，且价值总和最大。
    # 可以转换成01问题 和 完全背包的组合
    dp = [0]*(V+1)
    for i in range(1,N+1):
        ######################
        if cost[i-1]*num[i-1] >= V: # 物品个数x费用大于V，则认为有无限个
            for j in range(1, V + 1):
                if j >= cost[i - 1]:
                    dp[j] = max(dp[j], dp[j - cost[i - 1]] + value[i - 1])
                else:
                    dp[j] = dp[j]
            continue
        ########################
        iter = 1 # 接下来就是每个物品的数量划分成1，2，4，8等等来进行计算，可以化简为log时间
        counter = num[i-1]
        while (iter < counter ):
            # 对于每次都进行0-1背包算法, 每次的cost 和 value 都是* temp_count
            for j in range(V, -1, -1):
                if j >= iter*cost[i - 1]:
                    dp[j] = max(dp[j], dp[j - iter*cost[i - 1]] +  iter*value[i - 1])
                else:
                    dp[j] = dp[j]
            counter-= iter
            iter*=2
        # 对于减去1，2，4之后的进行01背包
        for j in range(V, -1, -1):
            if j >=  counter*cost[i - 1]:
                dp[j] = max(dp[j], dp[j - counter*cost[i - 1]] +  counter*value[i - 1])
            else:
                 dp[j] = dp[j]
    return dp[-1]


N = 8
V = 20
cost = [3,2,6,7,1,4,9,5]
value = [6,3,5,8,3,1,6,9]
num = [3,5,1,9,3,5,6,8]
print("01 knapsack is :", knapsack_01(N, V, cost, value))
print("01 knapsack using 1d space is :", knapsack_01_1d(N, V, cost, value))
print("complete knapsack is :", knapsack_complete(N, V, cost, value))
print("complete knapsack using 1d space is :", knapsack_complete_1d(N, V, cost, value))
print("multiple knapsack is :", knapsack_multi_1d(N, V, cost, value, num))



# 补充一道leetcode题目
# 32. 最长有效括号32. 最长有效括号
# 给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。
# 输入 ")()())"  输出4
def longestValidParentheses1( s) :
    # 用动态规划，难想到
    n = len(s)
    if n == 0:
        return 0
    dp = [0] * n
    ans = 0
    for i in range(1,n):
        # 只处理右括号
        if s[i] == ')':
            # 向前寻找对应的左括号
            pre = i-1 - dp[i-1];
            if pre>=0 and s[pre] == '(':
                dp[i] = dp[i-1] + 2
                #为了处理()()这种情况，也是需要将pre-1计算的加进去
                if pre>0:
                    dp[i] += dp[pre-1]
            ans = max(ans, dp[i])
    return ans;
def longestValidParentheses2( s) :
     # 不用动态规划； 用栈记录'(' , ')'的位置信息
    if not s:
        return 0
    ans = 0
    stack = [-1]
    for i in range(len(s)):
        if s[i] == "(":
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                ans = max(ans, i-stack[-1])
    return ans

# 补充两题：
# 53. 最大子序和
# 题目：给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
# 输入: [-2,1,-3,4,-1,2,1,-5,4],
# 输出: 6
# 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

# 152. 乘积最大子序列
# 题目：给定一个整数数组 nums ，找出一个序列中乘积最大的连续子序列（该序列至少包含一个数）。
# 输入: [2,3,-2,4]
# 输出: 6
# 解释: 子数组 [2,3] 有最大乘积 6。

def maxSubArray(nums):
    #动态规划 一维
    # dp[i] 是以i 结尾的最大值
    # dp[i]= max(dp[i-1]+nums[i],nums[i])
    if len(nums)==1:
        return nums[0]
    ans = [nums[0]] * len(nums)
    for i in range(1,len(nums)):
        ans[i]=max(ans[i-1]+nums[i],nums[i])
    return max(ans)

def maxProduct(nums):
    # 动态规划
    n = len(nums)
    max_all = -sys.maxsize - 1
    max_temp = [1]*(n+1)
    min_temp = [1]*(n+1)
    for i in range(1,n+1):
        if nums[i-1] < 0:
            max_temp[i] = max(min_temp[i-1]*nums[i-1],nums[i-1])
            min_temp[i] = min(max_temp[i-1]*nums[i-1],nums[i-1])
        else:
            max_temp[i] = max(max_temp[i-1]*nums[i-1],nums[i-1])
            min_temp[i] = min(min_temp[i-1]*nums[i-1],nums[i-1])
        max_all = max(max_temp[i],max_all)
    #print(max_temp)
    #print(min_temp)
    return max_all





