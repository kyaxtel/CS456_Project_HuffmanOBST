import heapq 

class HuffmanNode: # Node for Huffman tree
    def __init__(self, symbol, frequency):
        self.symbol = symbol    # byte or None
        self.frequency = frequency 
        self.left = None 
        self.right = None

    def __lt__(self, other): 
        return self.frequency < other.frequency # allows heapq to maintain the min-heap property based on frequency
    
def build_huffman_tree(frequency_table): # frequency_table: dict of symbol -> frequency
    priority_queue = [HuffmanNode(symbol, frequency) for symbol, frequency in frequency_table.items()]
    heapq.heapify(priority_queue) 

    # Build the tree
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue) 
        right = heapq.heappop(priority_queue) 
        merged_node = HuffmanNode(None, left.frequency + right.frequency) # internal node with combined frequency
        merged_node.left = left 
        merged_node.right = right 
        heapq.heappush(priority_queue, merged_node) # add the merged node back to the priority queue

    return priority_queue[0] # root of the tree 

def generate_huffman_codes(root): # root: root of the Huffman tree
    codes = {} 
    def generate_codes_helper(node, current_code): # recursive helper function to traverse the tree and generate codes
        if node is None:
            return
        
        # Leaf node -> assign code
        if node.symbol is not None:
            codes[node.symbol] = current_code
            return
        
        generate_codes_helper(node.left, current_code + '0')
        generate_codes_helper(node.right, current_code + '1')
    
    generate_codes_helper(root, '')
    return codes # return the mapping of symbols to their corresponding Huffman codes

def huffman_encode(data: bytes, code_map: dict) -> str: # data: input bytes to encode, code_map: dict of symbol -> Huffman code
    return ''.join(code_map[byte] for byte in data)

def huffman_decode(bitstring: str, root) -> bytes: # bitstring: the encoded string of '0's and '1's, root: root of the Huffman tree
    decoded_bytes = [] 
    current_node = root 
    for bit in bitstring: 
        current_node = current_node.left if bit == '0' else current_node.right 
        if current_node.symbol is not None: # reached a leaf
            decoded_bytes.append(current_node.symbol)
            current_node = root # reset to the root for the next symbol

    return bytes(decoded_bytes) # return the decoded bytes after traversing the bitstring through the Huffman tree