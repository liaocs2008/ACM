import random
import math
import datetime

root = {'wins':0, 'playouts':0, 'move':-1, 'child':[]}
T = 0 # number of simulations

def lose(board):
  # 0, 1, 2
  # 3, 4, 5
  # 6, 7, 8
  if board[0] != ' ':
    if (board[0] == board[1]) and (board[1] == board[2]): return True
    elif (board[0] == board[3]) and (board[3] == board[6]): return True
    elif (board[0] == board[4]) and (board[4] == board[8]): return True

  if board[1] != ' ':
    if (board[1] == board[4]) and (board[4] == board[7]): return True

  if board[2] != ' ':
    if (board[2] == board[4]) and (board[4] == board[6]): return True
    elif (board[2] == board[5]) and (board[5] == board[8]): return True

  if board[3] != ' ':
    if (board[3] == board[4]) and (board[4] == board[5]): return True

  if board[6] != ' ': 
    if (board[6] == board[7]) and (board[7] == board[8]): return True

  return False

def tellMoves(board, xo):
  moves = []
  for index in xrange(len(board)):
    if board[index] == ' ':
      # try if able to fill with 'x' or 'o'
      board[index] = xo
      if not lose(board): moves.append(index)
      board[index] = ' '
      #print board, moves
  #print " !!"
  return moves

def selectByUCT(nodes):
  return sorted(nodes, key=lambda n: 
      n['wins']/n['playouts'] + math.sqrt(16 * math.log(T)/ n['playouts']))[-1]

time_limit = 625 # milliseconds
elapsed_time = 0
timer = datetime.datetime.now()
while elapsed_time < time_limit:
  ME = 'x'
  xo = 'x' 
  board = [' '] * 9
  path = []

  # select a path to a leaf
  node = root
  while len(node['child']) == len(board)-len(path): # fully expanded
    node = selectByUCT(node['child'])
    board[node['move']] = xo
    path.append(node['move'])
    xo = 'x' if 'o' == xo else 'o'
  
  tmp_node = node
  valid_moves = tellMoves(board, xo)
  while len(path) < len(board) and valid_moves: 
    move = random.choice(valid_moves)

    # expand leaf to choose a child
    new_child = {}
    found = False
    for child in tmp_node['child']:
      if child['move'] == move:
        found = True
        new_child = child
        break
    if not found : new_child = {'wins':0, 'playouts':0, 'move':-1, 'child':[]}
    
    # simulate random play
    new_child['move'] = move
    board[move] = xo
    path.append(move)
    
    tmp_node['child'].append(new_child)
    tmp_node = new_child

    xo = 'x' if 'o' == xo else 'o'
    valid_moves = tellMoves(board, xo)
  

  # update along path
  #print board, path
  win = True if len(path) < len(board) and xo != ME else False
  node = root
  while node['child']:
    if len(path) < 1: print path, node['child']
    move = path.pop(0)
    for child in node['child']:
      if child['move'] != move: continue
      else :
        if win: child['wins'] += 1
        child['playouts'] += 1
        node = child
        break
  T += 1
  assert root['move'] == -1 and root['playouts'] == 0 and root['wins'] == 0
  elapsed_time = (datetime.datetime.now() - timer).microseconds / 1000

node = root
for child in node['child']:
  print child['wins'], 
print ""
for child in node['child']:
  print child['playouts'], 


def DFS_print(node, indent, level):
  if (level > 4): return
  indentString = ' ' * indent
  for child in node['child']:
    print indentString + "|-(" + str(child['wins']) + ',' +  str(child['playouts'])+")"
    DFS_print(child, indent + 6, level + 1)

#DFS_print(root, 1, 0)
