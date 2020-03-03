class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BinarySearchTree(object):
    def __init__(self, root = None ):
        self.root = root
    def insertNode(self, root, key):
        if not root:
            return Node(key)

        if key > root.val:
            root.right = self.insertNode(root.right, key)
        else:
            root.left = self.insertNode(root.left, key)
        return root

    def deleteNode(self, root, key):
        if not root:
            return None
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.right and not root.left:
                root = None
            elif root.left: #  统一先判断左节点
                node = root.left
                while node.right:
                    node = node.right
                root.val = node.val
                root.left = self.deleteNode(root.left,node.val)
            elif root.right:
                node = root.right
                while node.left:
                    node = node.left
                root.val = node.val
                root.right = self.deleteNode(root.right,node.val)
        return  root















class BinaryTree(object):
    def __init__(self, root = None):
        self.root = root

    # 建立二叉树是以层序遍历方式输入，节点不存在时以 'None' 表示
    # 例如 [0,1,2,None,4,5,6]
    def creatTree(self, nodeList):
        if len(nodeList) != 0:
            self.root = Node(nodeList[0])
            temp_list = [self.root]
            for i in range(1,len(nodeList)):
                item = nodeList[i]
                if item != None:
                    new_node = Node(item)
                    temp_list.append(new_node)
                    if i % 2 != 0 :
                        temp_list[i//2].left = new_node
                    else:
                        temp_list[i//2-1].right = new_node
            return self.root
        else:
            return None

    def LevelOrder(self, root):
        # 层序遍历
        queue_list = [root]  # 使用queue模拟
        ans = []
        while queue_list:
            node = queue_list.pop(0)
            if not node:
                continue
            ans.append(node.val)
            queue_list.append(node.left)
            queue_list.append(node.right)
        print(ans)
        return ans

    def PreOrder(self, root):
        # 先序遍历 递归
        if root is None:
            return
        print(root.val)
        self.PreOrder(root.left)
        self.PreOrder(root.right)

    def PreOrder_stack(self, root):
        # 先序遍历 非递归
        if root is None:
            return

        stack = []
        ans = []
        node = root
        while node or stack:
            # 从根节点开始，一直找它的左子树
            while node:
                ans.append(node.val)
                stack.append(node)
                node = node.left
            # while结束表示当前节点node为空，即前一个节点没有左子树了
            node = stack.pop()
            # 开始查看它的右子树
            node = node.right
        return ans


    def InOrder(self, root):
        # 中序遍历 递归
        if root is None:
            return
        self.InOrder(root.left)
        print(root.val)
        self.InOrder(root.right)

    def InOrder_stack(selfself, root):
        # 中序遍历 非递归
        if root is None:
            return
        stack = []
        ans = []
        node = root
        while node or stack:
            # 从根节点开始，一直找它的左子树
            while node:
                stack.append(node)
                node = node.left
            # while结束表示当前节点node为空，即前一个节点没有左子树了
            node = stack.pop()
            ans.append(node.val)
            # 开始查看它的右子树
            node = node.right
        return ans

    def PostOrder(self, root):
        # 后序遍历 递归
        if root is None:
            return
        self.PostOrder(root.left)
        self.PostOrder(root.right)
        print(root.val)

    def PostOrder_stack(self, root):
        # 后序遍历 非递归
        if root is None:
            return

        stack1 = []
        ans = []
        node = root
        stack1.append(node)
        # 这个while循环的功能是找出后序遍历的逆序，存在stack2里面
        while stack1:
            node = stack1.pop()
            if node.left:
                stack1.append(node.left)
            if node.right:
                stack1.append(node.right)
            ans.append(node.val)
        # 将stack2中的元素出栈，即为后序遍历次序
        return ans[::-1]


    def BuildTree_pre_in(self, preorder, inorder):
        # 前序和中序构建二叉树
        # preorder [0, 1, 3, 4, 2, 5, 6]
        # inorder  [3, 1, 4, 0, 5, 2, 6]
        if not preorder:
            return None
        root = Node(preorder[0])

        i = inorder.index(root.val)
        root.left = self.BuildTree_pre_in(preorder[1:i + 1], inorder[:i])
        root.right = self.BuildTree_pre_in(preorder[i + 1:], inorder[i+1:])
        return root



    def BuildTree_post_in(self, postorder, inorder):
        # 后序和中序构建二叉树
        # postorder [3, 4, 1, 5, 6, 2, 0]
        # inorder   [3, 1, 4, 0, 5, 2, 6]
        if not inorder:
            return None
        root = Node(postorder[-1])
        i = inorder.index(root.val)
        root.left = self.BuildTree_post_in(postorder[:i], inorder[:i])
        root.right = self.BuildTree_post_in(postorder[i:-1], inorder[i+1:])

        return root

    def BuildTree_pre_post(self, preorder, postorder):
        # 前序和后序构建二叉树 ， 答案不唯一
        # preorder  [0, 1, 3, 4, 2, 5, 6]
        # postorder [3, 4, 1, 5, 6, 2, 0]
        if not preorder:
            return None
        node = Node(preorder[0])
        if len(preorder) == 1:
            return node
        i = postorder.index(preorder[1])

        node.left = self.BuildTree_pre_post(preorder[1:i + 2],  postorder[:i + 1])
        node.right = self.BuildTree_pre_post(preorder[i + 2:],  postorder[i + 1:-1])

        return node


    def BuildTree_level_in(self, levelorder, inorder):
        # 层序和中序构建二叉树
        # levelorder [0, 1, 2, 3, 4, 5, 6]
        # inorder    [3, 1, 4, 0, 5, 2, 6]
        if  not inorder:
            return None
        # Check if that element exist in level order
        for i in range(0, len(levelorder)):
            if levelorder[i] in inorder:
                node = Node(levelorder[i])
                index = inorder.index(levelorder[i])
                break
        # Construct left and right subtree
        node.left = self.BuildTree_level_in(levelorder, inorder[0:index])
        node.right = self.BuildTree_level_in(levelorder, inorder[index + 1:len(inorder)])
        return node



if __name__ == '__main__':
    a = [20,8,22,4,12,11,None,None,None,10,14]
    #preorder   = [0, 1, 3, 4, 2, 5, 6]
    #inorder    = [3, 1, 4, 0, 5, 2, 6]
    #postorder  = [3, 4, 1, 5, 6, 2, 0]
    #levelorder = [0, 1, 2, 3, 4, 5, 6]
    Tree = BinaryTree()
    root_node = Tree.creatTree(a)

    # 验证二叉搜索树
    SearchTree = BinarySearchTree()
    root_node_search = None
    root_node_search = SearchTree.insertNode(root_node_search,8)
    root_node_search = SearchTree.insertNode(root_node_search, 5)
    root_node_search = SearchTree.deleteNode(root_node_search, 5)
    root_node_search = SearchTree.deleteNode(root_node_search, 8)
    Tree.InOrder(root_node_search)

    '''
    # 验证中序
    Tree.InOrder(root_node)
    print(Tree.InOrder_stack(root_node))
    # 验证前序
    Tree.PreOrder(root_node)
    print(Tree.PreOrder_stack(root_node))
    # 验证后序
    Tree.PostOrder(root_node)
    #print(Tree.PostOrder_stack(root_node))
    '''
    '''
    # 验证构建树
    preorder   = Tree.PreOrder_stack(root_node)
    inorder    = Tree.InOrder_stack(root_node)
    postorder  = Tree.PostOrder_stack(root_node)
    levelorder = [x for x in a if x != None  ]
    print(Tree.InOrder_stack(  Tree.BuildTree_pre_in(preorder, inorder)   ))
    print(Tree.InOrder_stack(  Tree.BuildTree_post_in(postorder, inorder)   ))
    print(Tree.InOrder_stack(  Tree.BuildTree_pre_post(preorder, postorder)   ))
    print(Tree.InOrder_stack(  Tree.BuildTree_level_in(levelorder, inorder)))
    '''

