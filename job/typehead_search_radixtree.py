#!/usr/bin/env python
# This failed the two largest data set, timeout
# http://codereview.stackexchange.com/questions/24602/readability-and-performance-of-my-trie-implementation

import datetime
from collections import defaultdict

class Node(dict):
  def __init__(self):
    self.id = []
 
class Trie:
  def __init__(self):
    self.root = Node()

  def search(self, string, new=False):
    node = self.root
    for ch in string:
      if ch in node.keys():
        node = node[ch]
      elif new:
        new_node = Node()
        node[ch]=  new_node
        node = new_node
      else : 
        raise KeyError
    return node

  def list(self, pre=''):
    try :
      node = self.search(pre)
      stack = [(pre, node)]
      while stack:
        tmp_pre, node = stack.pop()
        if node.id:
          yield tmp_pre, node.id
        for ch, child in node.items():
          stack.append((tmp_pre + ch, child))
    except KeyError:
      yield '', []

  def prefix(self, pre=''):
    l = []
    for string, ids in self.list(pre):
      l += ids
    return set(l)

  def insert(self, string, id):
    node = self.search(string, new=True)
    node.id += [id]
    node.leaf_label = True

  def remove(self, string, id):
    try:
      node = self.search(string)
      if node.id:
        node.id.remove(id)
    except KeyError:
      return

if __name__ == "__main__":
  tree = Trie()
  #tree.insert("ab", "12")
  #tree.insert("ac", "12")
  #tree.insert("ac", "12")
  N = input()
  database = {}
  for i in xrange(0, N):
  #while True:
    command = raw_input().split()
    if "ADD" == command[0]:
      (type, id, score, string) = (command[1], command[2], float(command[3]), map(lambda x:x.lower(), command[4:]))
      database[id] = {'string':string, 'score': score, 'type':type, 'time':datetime.datetime.now()}
      for s in string: tree.insert(s, id)
    elif "DEL" == command[0]:
      id = command[1]
      if id in database.keys():
        for s in database[id]['string']:
          tree.remove(s, id)
        del database[id]
    elif "QUERY" == command[0]:
      (num, tokens) = (int(command[1]), map(lambda x:x.lower(), command[2:]))
      ids = tree.prefix(tokens[0])
      for token in tokens[1:]:
        ids = ids.intersection(tree.prefix(token))
      if not ids: print ''
      else : 
        result = zip(*sorted(map(lambda id: (database[id], id), ids), 
                                  key=lambda p: (p[0]['score'], p[0]['time']), 
                                  reverse=True))[1][:num]
        print (' '.join([r for r in result]))
    elif "WQUERY" == command[0]:
      (num, num_boost) = (int(command[1]), int(command[2]))
      id_boost = defaultdict(lambda : 1)
      type_boost = {'user':1, 'topic':1, 'question':1, 'board':1}
      for i in xrange(num_boost):
        (boost, factor) = command[3 + i].split(':')
        if boost in type_boost.keys(): type_boost[boost] = float(factor)
        else : id_boost[boost] = float(factor)
      tokens = map(lambda x: x.lower(), command[(3 + num_boost):])
      ids = tree.prefix(tokens[0])
      for token in tokens[1:]:
        ids = ids.intersection(tree.prefix(token))
      if not ids: print ''
      else : 
        result = zip(*sorted(map(lambda id: (database[id], id), ids), 
                             key=lambda p: (p[0]['score']*type_boost[p[0]['type']]*id_boost[p[1]], p[0]['time']), 
                             reverse=True))[1][:num]
        print (' '.join([r for r in result]))
