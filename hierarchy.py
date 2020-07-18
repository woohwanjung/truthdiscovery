
# coding: utf-8

# In[2]:

#%reset
import random




class Node(object):
    def __init__(self, h, parent, name):
        self.depth = 0
        self.children = []
        self.name = name
        self.rootid = 0
                        
        self.parent = parent
        h.size += 1
        h.regiName(self)
        if parent:
            self.depth = parent.depth+1
            self.rootid = parent.rootid
            parent.addChild(self)
        self.h = h
    
    def computeHeight(self):
        if len(self.children)==0:
            return 0
        height = 0
        
        for child in self.children:
            height = max(height, child.computeHeight()+1)
        return height
        
        
    def traversal(self, node_list, parent_id):
        nid = len(node_list)
        node_list.append((nid, self.name, parent_id))
        
        for child in self.children:
            child.traversal(node_list, nid)
    
    def addChild(self,child):
        self.children += [child]
    
    def countNodes(self, LeafOnly = False):
        if LeafOnly and len(self.children)>0:
            c = 0
        else:
            c = 1

        for child in self.children:
            c+=child.countNodes(LeafOnly)
        return c
    
    
    def setMaxDepth(self):
        if self.children:
            for child in self.children:
                child.setMaxDepth()
        else:
            self.h.max_depth = max(self.h.max_depth, self.depth)
    
    
    def printNode(self):
        out = ""
        for i in range(self.depth):
            out += "\t"
        try:
            print (out, self.name)
        except UnicodeEncodeError:
            print(self.name.encode("utf-8"))
        for child in self.children:
            child.printNode()
    
    def isAncestorof(self,ans):
        if self == ans:
            return True
        if self.rootid != ans.rootid:
            return False
        if self.depth  > ans.depth: 
            return False
        
        node = ans
        for i in range(ans.depth - self.depth):
            node = node.parent
        
        return node == self

    def printAncestor(self):
        if self.parent is not None:
            self.parent.printAncestor()
            print(self.parent.__str__())
    
    def setDepth(self, d):
        self.depth = d
        for child in self.children:
            child.setDepth(d+1)
        
    
    def __str__(self):
        out = "Name: %s, Root: %d, Depth: %d"%(self.name,self.rootid,self.depth)
        return out
    
    def __repr__(self):
        return "("+self.__str__()+")"
    
 
     
class Hierarchy(object):
    def __init__(self):
        self.roots = []
        self.nodes = []
        self.size = 0
        self.name2nodes = {}
        self.max_depth = 0
        self.preprocessed = False
        
        
    def build(self, dat):
        for nid, name, pid in dat:
            if nid != len(self.nodes):
                print("Invalid data - ")
            
            if pid <0:
                node = Node(self, None, "Trivial Root")
            elif pid == 0:
                node = Node(self, None, name)
                rootid = len(self.roots)
                node.rootid = rootid
                self.roots.append(node)
            else:
                parent = self.nodes[pid]
                node = Node(self, parent, name)
                
            self.nodes.append(node)


    def traversal(self):
        node_list = [[0,"Root",-1]]
        for root in self.roots:
            root.traversal(node_list, 0)
        return node_list
        
    
    
    def computeHeight(self):
        height = 0
        for root in self.roots:
            height = max(height, root.computeHeight()+1)
        return height
        
    def getNodes(self, name):
        try:
            return self.name2nodes[name]
        except KeyError:
            return None

    
    def findingMatchingNodesPreprocessed(self, e, claims):
        candidates = [self.getNodes(claim[0]) for claim in claims]
        matched_nodes = []
        for c in range(len(candidates)):
            for cand in candidates[c]:
                if cand.name == claims[c][0]:
                    matched_nodes.append(cand)
                    break
            
        if len(matched_nodes) != len(claims):
            print("Matching Error")
        
        return matched_nodes
    
    
    def findingMatchingNodes(self, claims):
        matching_nodes = []
        if len(claims) == 0:
            return matching_nodes
        if len(claims) == 1:
            name = claims[0][0]
            matching_nodes.append(random.choice(self.getNodes(name)))
            return matching_nodes
            
        #claims = sorted(claims, key=lambda claim: claim[1], reverse = True)
        
        candidates = [self.getNodes(claim[0]) for claim in claims]
        matched_nodes = []
        
        
        
        #Find a matching node for the first node
        maxPair = None
        maxDepthSum = -1
        AncestorDescedentPair = False
        
        for cand0 in candidates[0]:
            for cand1 in candidates[1]:
                depthSum = cand0.depth + cand1.depth

                if AncestorDescedentPair:
                    if depthSum <= maxDepthSum:
                        continue
                    if cand0.isAncestorof(cand1) or cand1.isAncestorof(cand0):
                        maxDepthSum = depthSum
                        maxPair = (cand0, cand1)
                else:
                    if cand0.isAncestorof(cand1) or cand1.isAncestorof(cand0):
                        maxDepthSum = depthSum
                        maxPair = (cand0, cand1)
                        AncestorDescedentPair  = True
                    else:
                        if depthSum >= maxDepthSum:
                            maxDepthSum = depthSum
                            maxPair = (cand0, cand1)                            
                            
        matched_nodes.append(maxPair[0]) 
        matched_nodes.append(maxPair[1]) 
        
        
        for e in range(2,len(candidates)):
            if candidates[e] == None:
                print("Err")
            if len(candidates[e]) == 1:
                matched_nodes.append(candidates[e][0])
                continue
            maxCount = -1
            maxNode = None
            for cand in candidates[e]:
                count = 0
                for matched_node in matched_nodes:
                    if cand.isAncestorof(matched_node) or matched_node.isAncestorof(cand):
                        count+=1
                if count > maxCount:
                    maxCount = count
                    maxNode = cand
            matched_nodes.append(maxNode)
        
        return matched_nodes
        
                
    def _regiName(self, node, name):
        try:
            nodes = self.name2nodes[name]
            nodes.append(node)
        except KeyError:
            nodes = [node]
            self.name2nodes[name] = nodes
            
    def regiName(self, node):
        self._regiName(node, node.name)
            
            
    def computeDistance(self,node1,node2):
        if node1.rootid != node2.rootid:
            distance = 2+ node1.depth +node2.depth
            return distance
        distance = 0
        #FOR DEBUG
        try:
            while node1 != node2:
                if node1.depth > node2.depth:
                    node1 = node1.parent
                else:
                    node2 = node2.parent
                distance+=1
        except AttributeError as AE:
            raise AE
        return distance
    
    def computeMinimumDistance(self, node_from, nodes_to):
        minDist = 1000
        minNode = None
        
        for nt in nodes_to:
            dist = self.computeDistance(node_from, nt)
            if dist < minDist:
                minDist = dist
                minNode  = nt
                if minDist == 0:
                    break
                
        return (dist,minNode)
# In[21]:
    
    def setMaxDepth(self):
        self.max_depth = 0
        for root in self.roots:
            root.setMaxDepth()
            
    def printH(self):
        for root in self.roots:
            root.printNode()

if __name__ == "__main__":
    h = Hierarchy()
    
    dat = [[0,"Root",-1],[1,"Europe",0],[2,"Albania",1],[3,"County in Al",2],[4,"Asia",0],[5,"Korea",4],[6,"Japan",4]]
    h.build(dat)
    
    h.printH()

    
