# 背诵 排序

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
# 创建堆
def build_heap(lists, size):
    for i in range((size >> 1),-1,-1):
        max_heap(lists, i, size)
# 堆排序
def heap_sort(nums):
    n = len(nums)
    build_heap(nums, n) # 建堆
    for i in range(n-1, 0, -1): # 维护堆，把最大值放大最后去
        nums[0], nums[i] = nums[i], nums[0]
        max_heap(nums, 0, i)
    return nums








nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result1 = bubble_sort(nums)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result2 = select_sort(nums)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result3 = insert_sort(nums)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result4 = quick_sort(nums,0, len(nums)-1)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result5 = merge_sort(nums)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result6 = heap_sort(nums)
nums = [54, 26, 93, 17, 77, 31, 44, 55, 20]
result7 = shell_sort(nums)
print(result1)
print(result2)
print(result3)
print(result4)
print(result5)
print(result6)
print(result7)