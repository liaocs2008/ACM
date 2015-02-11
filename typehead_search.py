#!/usr/bin/env python
import datetime

def search(strings, token, strings_pos = None):
  if not strings_pos: 
    strings_pos = []
    for i in xrange(len(strings)):
      check = False
      #print i, strings[i]
      for s in strings[i]:
        if token in s and token[0] == s[0]: 
          check = True
          break
        elif token < s: break
      if check: strings_pos.append(i)
    return strings_pos
  else:
    tmp_strings_pos = []
    for i in xrange(len(strings_pos)):
      check = False
      for s in strings[strings_pos[i]]:
        if token in s and token[0] == s[0]:
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
  pos = 0
  for i in xrange(0, N):
    command = raw_input().split(' ')
    if command[0] == "ADD":
      (type, id, score, string) = (command[1], command[2], float(command[3]), sorted(map(lambda x:x.lower(), command[4:])))
      database[id] = {'score': score, 'type':type, 'pos':pos, 'time':datetime.datetime.now()}
      strings.append(string)
      ids.append(id)
      pos += 1
    elif command[0] == "DEL":
      id = command[1]
      string_pos = database[id]['pos']
      del strings[string_pos]
      del ids[string_pos]
      database.pop(id)
      pos -= 1
    elif command[0] == "QUERY":
      (num, tokens) = (int(command[1]), map(lambda x:x.lower(), command[2:]))
      strings_pos = search(strings, tokens[0])
      #print strings_pos
      for token in tokens[1:]:
        #print strings_pos
        strings_pos = search(strings, token, strings_pos)
        if not strings_pos: break
      if strings_pos:
        records = [database[ids[i]] for i in strings_pos]
        #print records
        #print strings_pos
        ids_pos = zip(*sorted(zip(records, strings_pos), key=lambda pair: (pair[0]['score'], pair[0]['time']), reverse=True))[1][:num]
        print (' '.join([ids[i] for i in ids_pos]))
      else: print ''

  #print database
  #print strings
  #print ids
  #print pos
