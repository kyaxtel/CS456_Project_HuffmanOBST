class OBSTNode: # Optimal Binary Search Tree Node
    def __init__(self, key, value = None):
        self.key = key # key of the node
        self.value = value # associated value (optional, can be used for storing data)
        self.left = None
        self.right = None

def build_obst(keys, probabilities, values = None): # keys: list of keys, probabilities: list of probabilities for each key, values: optional list of values corresponding to keys
    n = len(keys)

    # DP tables (1-based indexing for convenience)
    e = [[0]*(n+2) for _ in range(n+2)] # expected cost
    w = [[0]*(n+2) for _ in range(n+2)] # probability sum
    root_table = [[0]*(n+2) for _ in range(n+2)] 

    # Initialize w[i][i-1] = 0 (empty trees)
    for i in range(1, n+2):
        w[i][i-1] = 0

    # Fill w[i][j] = sum of probabilities[i-1..j-1]
    for i in range(1, n+1):
        w[i][i] = probabilities[i-1]
        for j in range(i+1, n+1):
            w[i][j] = w[i][j-1] + probabilities[j-1]

    # DP computation
    for l in range(1, n+1):  # length of subtree
        for i in range(1, n-l+2):  # start index
            j = i + l - 1  # end index
            e[i][j] = float('inf')
            for r in range(i, j+1):
                left = e[i][r-1] if r-1 >= i else 0
                right = e[r+1][j] if r+1 <= j else 0
                cost = left + right + w[i][j]
                if cost < e[i][j]:
                    e[i][j] = cost
                    root_table[i][j] = r

    # Function to recursively build OBST
    def build_tree(i, j):
        if i > j:
            return None
        r = root_table[i][j]
        node = OBSTNode(keys[r-1], values[r-1] if values else None)
        node.left = build_tree(i, r-1)
        node.right = build_tree(r+1, j)
        return node
    
    return build_tree(1, n) # return the root of the constructed OBST
    

def obst_search(root: OBSTNode, key): # Search for a key in the OBST and count comparisons
    comparisons = 0
    node = root
    while node is not None:
        comparisons += 1
        if key == node.key:
            return node.value, comparisons
        elif key < node.key:
            node = node.left
        else: 
            node = node.right
    return None, comparisons # return None if key not found, along with the number of comparisons made