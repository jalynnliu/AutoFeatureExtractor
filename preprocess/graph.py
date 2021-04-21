import pandas as pd
from util import log, timeclass

class Graph:
    def __init__(self,info,tables):
        
        self.info = info
        
        self.table2info = info['tables']
        self.relations = info['relations']

    @timeclass(cls='Graph')
    def dfs(self,root_name, graph, depth):
        '''
        inspired by Featuretools, instead of a search algorithm, DFS here means to aggregate columns defined by depth.
        different from featuretools which let users define aggregation function, graph.dfs is fully automatic.
        '''
        depth[CONSTANT.MAIN_TABLE_NAME]['depth'] = 0
        queue = deque([root_name])
        while queue:
            u_name = queue.popleft()
            for edge in graph[u_name]:
                v_name = edge['to']
                if 'depth' not in depth[v_name]:
                    depth[v_name]['depth'] = depth[u_name]['depth'] + 1
                    queue.append(v_name)

    @timeclass(cls='Graph')
    def build_depth(self):
        rel_graph = defaultdict(list)
        depth = {}
        
        for tname in self.tables:
            depth[tname] = {}
            
        for rel in self.relations:
            ta = rel['table_A']
            tb = rel['table_B']
            rel_graph[ta].append({
                "to": tb,
                "key": rel['key'],
                "type": rel['type']
            })
            rel_graph[tb].append({
                "to": ta,
                "key": rel['key'],
                "type": '_'.join(rel['type'].split('_')[::-1])
            })
        self.dfs(CONSTANT.MAIN_TABLE_NAME, rel_graph, depth)
        
        self.rel_graph = rel_graph
        self.depth = depth