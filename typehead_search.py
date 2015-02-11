#!/usr/bin/env python
import datetime
from collections import defaultdict

def search(strings, token, strings_pos = None):
  length = len(token)
  if not strings_pos:
    #print "NONE STRING POS"
    strings_pos = []
    for i in xrange(len(strings)):
      check = False
      #print i, strings[i]
      for s in strings[i]:
        #if token[0] == s[0] and token in s: 
        if token[:length] == s[:length]:
          check = True
          break
        elif token < s: break
      if check: strings_pos.append(i)
    return strings_pos
  else :
    #print "FOUND STRING POS", strings_pos
    tmp_strings_pos = []
    for i in xrange(len(strings_pos)):
      check = False
      for s in strings[strings_pos[i]]:
        #if token[0] == s[0] and token in s:
        if token[:length] == s[:length]:
          #print "i=", i, "string=", strings[strings_pos[i]]
          check = True
          break
        elif token < s: break
      if check: tmp_strings_pos.append(strings_pos[i])
    return tmp_strings_pos

if __name__ == "__main__":
  database = {}
  strings = []
  ids = []
  N = input()
  for i in xrange(0, N):
  #while True:
    command = raw_input().split()
    #print "LISTEN: ", command
    if command[0] == "ADD":
      (type, id, score, string) = (command[1], command[2], float(command[3]), sorted(map(lambda x:x.lower(), command[4:])))
      database[id] = {'score': score, 'type':type, 'time':datetime.datetime.now()}
      strings.append(string)
      ids.append(id)
    elif command[0] == "DEL":
      id = command[1]
      if id in database.keys():
        database.pop(id)
        pos = ids.index(id)
        del strings[pos]
        del ids[pos]
    elif command[0] == "QUERY":
      (num, tokens) = (int(command[1]), map(lambda x:x.lower(), command[2:]))
      strings_pos = search(strings, tokens[0])
      #print strings_pos
      for token in tokens[1:]:
        if not strings_pos: break
        strings_pos = search(strings, token, strings_pos)
        #print strings_pos
      if strings_pos:
        records = [database[ids[i]] for i in strings_pos]
        #print records
        #print strings_pos
        ids_pos = zip(*sorted(zip(records, strings_pos), key=lambda pair: (pair[0]['score'], pair[0]['time']), reverse=True))[1][:num]
        print (' '.join([ids[i] for i in ids_pos]))
      else: print ''
    elif command[0] == "WQUERY":
      (num, num_boost) = (int(command[1]), int(command[2]))
      type_boost = {'user':1, 'topic':1, 'question':1, 'board':1}
      id_boost = defaultdict(lambda: 1)
      for i in xrange(num_boost):
        (boost, factor) = command[3 + i].split(':')
        if boost in type_boost.keys(): type_boost[boost] = float(factor)
        else : id_boost[boost] = float(factor)
      tokens = sorted(map(lambda x: x.lower(), command[(3 + num_boost):]))
      #print type_boost
      #print id_boost
      #print tokens
      strings_pos = search(strings, tokens[0])
      #print strings_pos
      for token in tokens[1:]:
        if not strings_pos: break
        strings_pos = search(strings, token, strings_pos)
      if strings_pos:
        #print "string_pos=", strings_pos
        records = [database[ids[i]] for i in strings_pos]
        #print records
        #print strings_pos
        ids_pos = zip(*sorted(zip(records, strings_pos), 
          key=lambda pair: (pair[0]['score']*type_boost[pair[0]['type']]*id_boost[ids[pair[1]]], pair[0]['time']), 
          reverse=True))[1][:num]
        print (' '.join([ids[i] for i in ids_pos]))
        #if '438' == (' '.join([ids[i] for i in ids_pos])):
          #print "LISTEN:", command
          #print database.keys()
          #print ids
      else: print ''
    #print database.keys()
    #print ids
  #print database
  #print strings
  #print ids
  #print pos
