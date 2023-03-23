import heapq as hq


class RelationSet:
    def __init__(self):
        self.heap = []
        self.set = set()

    def __len__(self):
        return len(self.set)
    
    def __print__(self):
        print('Relation set:')
        for relation in self.heap:
            print(f'\tRelation: {relation[1]}, Subj: {relation[0]}, Obj: {relation[2]}')

    def add(self, element, priority):
        if element not in self.set:
            hq.heappush(self.heap, (priority, element))
            self.set.add(element)

    def pop(self):
        priority, element = hq.heappop(self.heap)
        self.set.remove(element)
        return element
        
    def size(self):
        return len(self.set)