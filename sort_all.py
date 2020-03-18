# 背诵 排序

'''
  |  算法  |  平均复杂度  |  最好情况   |  最坏情况  | 空间复杂度  |  排序方式   | 稳定性|
  |  冒泡  |    O（n^2）  |   O(n)     |  O（n^2）  |    O(1)    |   in-place |   Y   |
  |  选择  |    O（n^2）  |   O（n^2） |  O（n^2）  |    O(1)    |    in      |   N   |
  |  插入  |    O（n^2）  |   O(n)     |  O（n^2）  |    O(1)    |    in      |   Y   |
  |  希尔  |    O(nlogn)  |   O(n)    |  O(n^2)    |    O(1)    |    in      |   N    |
  |  折半  |      O(n^2)(移动次数没变,和插入一样)    |   O(1}     |    in      |   Y    |
  |  快速  |   O (nlogn) |  O(nlogn)  |   O(n^2)   |   O(logn)  |    in      |   N    |
  |  堆排  |   O (nlogn) |  O(nlogn)  |   O(nlogn) |   O(1)     |    in      |   N    |
  |  归并  |   O (nlogn) |  O(nlogn)  |  O(nlogn)  |   O(n)     |    out     |   Y    |
  |  计数  |   O(n+k)    |  O(n+k)    |  O(n+k)    |   O(k)     |    out     |   Y    |  k为最大值
  |  桶排  |   O(n+k)    |  O(n+k)    |   O(n^2)   |   O(n+k)   |    out     |   Y    |
  |  基数  |   O(n*k)    |  O(n*k)    |  O(n*k)    |   O(n+k)   |    out     |   Y    |  k为基

'''
# https://www.zhihu.com/question/36738189
# https://www.jianshu.com/p/742d0a19d933
# https://www.jianshu.com/p/ff4bf4688ae0







# 冒泡排序
# 左右比较，然后最大的放在最后
def bubble_sort(nums):
    n = len(nums)
    for i in range(n):
        for j in range(0, n - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    return nums


# 选择排序
# 选择最小的放在最前面
def select_sort(nums):
    for i in range(len(nums)):
        min_idx = i
        for j in range(i + 1, len(nums)):
            if nums[j] < nums[min_idx] :
                min_idx = j
        nums[i], nums[min_idx] = nums[min_idx], nums[i]
    return nums

# 插入排序
# 保证小的都跑到左边去
def insert_sort(nums):
    for i in range(1, len(nums)):
        key = nums[i]
        j = i - 1
        while j >= 0 and key < nums[j]:
            nums[j + 1] = nums[j]
            j -= 1
        nums[j + 1] = key
    return nums

# 折半插入排序
# 插入的时候使用二分查找
def binaryinsert_sort(nums):
    for i in range(1, len(nums)):
        val = nums[i]
        low = 0
        high = i - 1
        while low <= high:
            mid = (low + high) // 2
            if val > nums[mid]:
                low = mid + 1
            else:
                high = mid - 1
        # 跳出循环后 low, mid 都是一样的, hight = low - 1
        for j in range(i, low, -1):
            nums[j] = nums[j - 1]
        nums[low] = val
    return nums


# 希尔排序
# 插入排序的一种，也称为缩小增量排序
# 举例 先通过 n/2的间隔插入排序， 再通过 n/4 ，知道1 的间隔排序
def shell_sort(nums):
    n = len(nums)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = nums[i]
            j = i
            while j >= gap and nums[j - gap] > temp:
                nums[j] = nums[j - gap]
                j -= gap
            nums[j] = temp
        gap //= 2
    return nums




# 快速排序
def partition(nums, low, high):
    i = (low - 1)
    pivot = nums[high]
    for j in range(low, high): # 将小于pivot的值都放在左边
        if nums[j] <= pivot:
            i = i + 1
            nums[i], nums[j] = nums[j], nums[i]
    nums[i + 1], nums[high] = nums[high], nums[i + 1]  # 替换pivot
    return (i + 1)
def quick_sort(nums, low, high):
    if low < high:
        pi = partition(nums, low, high)
        quick_sort(nums, low, pi - 1)
        quick_sort(nums, pi + 1, high)
    return nums

'''
# 快速排序简单写法 #但是会发现会使用额外内存
def quick_sort_simple(nums):
    if len(nums) <= 1:
        return nums
    pivot = nums[len(nums) // 2]
    left = [x for x in nums if x < pivot]
    middle = [pivot]
    right = [x for x in nums if x > pivot]
    return quick_sort_simple(left) + middle + quick_sort_simple(right)
'''
# 归并排序
# 原版
def merge(left, right):  # 合并两个有序数组
    l, r = 0, 0
    result = []
    while l < len(left) and r < len(right):
        if left[l] <= right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    result += left[l:]
    result += right[r:]
    return result


def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    num = len(nums) >> 1
    left = merge_sort(nums[:num])
    right = merge_sort(nums[num:])
    return merge(left, right)

'''
# 归并排序
# 这个是稍微改进的，避免在每次迭代的时候创建
temp = [0] * 100
def merge(nums, low, mid, high):
    i = low
    j = mid + 1
    size = 0
    while i <= mid and j <= high:
        if nums[i] < nums[j]:
            temp[size] = nums[i]
            i += 1
        else:
            temp[size] = nums[j]
            j += 1
        size += 1
    while i <= mid:
        temp[size] = nums[i]
        size += 1
        i += 1
    while j <= high:
        temp[size] = nums[j]
        size += 1
        j += 1
    for i in range(size):
        nums[low + i] = temp[i]


def merge_sort(nums, low, high):
    if low >= high:
        return
    mid = (low + high) >> 1
    merge_sort(nums, low, mid)
    merge_sort(nums, mid + 1, high)
    merge(nums, low, mid, high)
    return nums
'''

# 堆排
# 最大堆
def max_heap(nums, i, size):
    lchild = 2 * i + 1
    rchild = 2 * i + 2
    max = i
    if i < size / 2:
        if lchild < size and nums[lchild] > nums[max]:
            max = lchild
        if rchild < size and nums[rchild] > nums[max]:
            max = rchild
        if max != i:
            nums[max], nums[i] = nums[i], nums[max]
            max_heap(nums, max, size)
def build_heap(lists, size):
    for i in range((size >> 1),-1,-1):
        max_heap(lists, i, size)
def heap_sort(nums):
    n = len(nums)
    build_heap(nums, n) # 建堆
    for i in range(n-1, 0, -1): # 维护堆，把最大值放大最后去
        nums[0], nums[i] = nums[i], nums[0]
        max_heap(nums, 0, i)
    return nums


# 计数排序
# 使用一个额外数组counter在 counter[num] 记录出现的次数
def counting_sort(nums, k):  # k = max(a)
    n = len(nums)  # 计算a序列的长度
    output = [0 for i in range(n)]  # 设置输出序列并初始化为0
    counter = [0 for i in range(k + 1)]  # 设置计数序列并初始化为0，
    for j in nums:
        counter[j] = counter[j] + 1
    for i in range(1, len(counter)):
        counter[i] = counter[i] + counter[i-1]
    for j in nums:
        output[counter[j] - 1] = j
        counter[j] = counter[j] - 1
    return output

# 桶排序
# 将输入分段，然后每一段单独排序
def bucket_sort(nums, step):
    nums_min = min(nums)
    nums_max = max(nums)
    bucket_count = nums_max // step - nums_min // step + 1  # 获取桶的个数
    bucket_lists = [[] for _ in range(bucket_count)]  # 桶数组
    # 将值分配到桶中
    for i in nums:
        bucket_index = (i-nums_min) // step  # 获取每个元素所在的桶的索引值
        bucket_lists[bucket_index].append(i)
    # 每个桶内进行排序
    for bucket_list in bucket_lists:
        bucket_list = insert_sort(bucket_list)
    # 组合每个桶的元素
    result_list = []
    for bucket_list in bucket_lists:
        if len(bucket_list) != 0:
            result_list.extend(bucket_list)
    return result_list

# 基数排序
# 每一位进行counting sort
def radix_sort(s):
    i = 0 # 记录当前正在排拿一位，最低位为1
    max_num = max(s)  # 最大值
    j = len(str(max_num))  # 记录最大值的位数
    while i < j:
        bucket_list =[[] for _ in range(10)] #初始化桶数组
        for x in s:
            bucket_list[int(x / (10**i)) % 10].append(x) # 找到位置放入桶数组
        #print(bucket_list)
        s.clear()
        for x in bucket_list:   # 放回原序列
            for y in x:
                s.append(y)
        i += 1
    return s


nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result1 = bubble_sort(nums)
print(result1)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result2 = select_sort(nums)
print(result2)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result3 = insert_sort(nums)
print(result3)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result4 = shell_sort(nums)
print(result4)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result5 = binaryinsert_sort(nums)
print(result5)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result6 = quick_sort(nums,0, len(nums)-1)
print(result6)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result7 = merge_sort(nums)
print(result7)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result8 = heap_sort(nums)
print(result8)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result9 = counting_sort(nums,max(nums))
print(result9)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result10 = bucket_sort(nums,4)
print(result10)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result11 = radix_sort(nums)
print(result11)