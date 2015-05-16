#include "base.hpp"

#include <vector>
#include <algorithm>
#include <iomanip>

class BlockNode : public Node {
  public:
    BlockNode() {}
    BlockNode(const vector<int>& board) : Node()
    {
      board_.resize(board.size());
      copy(board.begin(), board.end(), board_.begin());
      unique_hash_ = 0;
      for(auto& i : board_) {
        unique_hash_ ^= i + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);
      }
    }
    ~BlockNode() {}

    void show()
    {
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          cout << std::setw(3) << board_[i*6 + j] << ",";
        }
        cout << endl;
      }
      cout << "depth=" << depth_  << ", c_cost=" << current_cost_ << ", f_cost=" << future_cost_ << endl;
    }

    vector<int> board_;
};

class BlockMove : public Move {
  public:
    BlockMove(int block_id = -1, int block_dst = -1) {block_id_ = block_id; block_dst_ = block_dst;}
    ~BlockMove() {}
    int block_id_;
    int block_dst_;
};

class Block : public Problem {
  public:
    Block(const string& name, int width, int height,
        const vector<int>& id, const vector<int>& length, const vector<int>& direction,
        const vector<int>& board, int target_id)
      : block_id_(id), block_length_(length), block_move_direction_(direction)
    {
      // be careful, initialize width and height before calling get_future_cost_
      target_id_ = target_id;
      width_ = width;
      height_ = height;
      name_ = name;

      init_ = make_shared<BlockNode>(board);
      init_->parent_move_ = make_shared<BlockMove>();
      init_->future_cost_ = this->get_future_cost(init_);

      // no need to initialize goal state, you can't specify goal node
      // any node with target block out of board is goal node
    }
    ~Block() {}

    inline void insert_new_node(const node_ptr&u, node_ptr& new_node, shared_ptr<node_set>& l)
    {
      new_node->parent_ = u;
      //new_node->parent_move_ = make_shared<PuzzleMove>(dynamic_pointer_cast<PuzzleNode>(new_node)->blank_);
      new_node->depth_ = u->depth_ + 1;
      new_node->current_cost_ = u->current_cost_ + new_node->parent_move_->cost();
      new_node->future_cost_ = this->get_future_cost(new_node);
      l->insert(new_node);
    }

    void get_successor(const node_ptr& u, shared_ptr<node_set>& l)
    {
      assert(true == l->empty());

      for (vector<int>::iterator it = block_id_.begin();
            it != block_id_.end();
            ++it)
      {
        const int id = *it;
        const int len = block_length_[id];
        // pos is the left up corner
        const int pos = find(dynamic_pointer_cast<BlockNode>(u)->board_.begin(),
                             dynamic_pointer_cast<BlockNode>(u)->board_.end(),
                             id) - dynamic_pointer_cast<BlockNode>(u)->board_.begin();
        const int row = pos / width_, col = pos % width_;

        if (block_move_direction_[id] > 0) // move horizontally
        {
          if (col + len-1 < width_-1) {
            int new_pos = row * width_ + col + 1; // move right
            if (dynamic_pointer_cast<BlockNode>(u)->board_[new_pos+len-1] == -1) {
              vector<int> board(dynamic_pointer_cast<BlockNode>(u)->board_);
              board[pos] = -1; // move, empty
              board[new_pos] = id;
              board[new_pos+len-1] = id;
              node_ptr new_node = make_shared<BlockNode>(board);
              new_node->parent_move_ = make_shared<BlockMove>(id, new_pos);
              insert_new_node(u, new_node, l);
            }
          } else if (id == target_id_) { // solution found
            int new_pos = row * width_ + col;
            vector<int> board(dynamic_pointer_cast<BlockNode>(u)->board_);
            for (int i = 0; i < len; ++i) {
              board[new_pos + i] = -1;
            }
            node_ptr new_node = make_shared<BlockNode>(board);
            new_node->parent_move_ = make_shared<BlockMove>(id, new_pos+len-1);
            insert_new_node(u, new_node, l);
          }

          if (col - 1 >= 0) {
            int new_pos = row * width_ + col - 1; // move left
            if (dynamic_pointer_cast<BlockNode>(u)->board_[new_pos] == -1) {
              vector<int> board(dynamic_pointer_cast<BlockNode>(u)->board_);
              board[pos + len-1] = -1; // move, empty
              board[new_pos] = id;
              board[new_pos+len-1] = id;
              node_ptr new_node = make_shared<BlockNode>(board);
              new_node->parent_move_ = make_shared<BlockMove>(id, new_pos);
              insert_new_node(u, new_node, l);
            }
          }
        }
        else // move vertically
        {
          if (row + len < height_) {
            int new_pos = (row + 1) * width_ + col; // move down
            if (dynamic_pointer_cast<BlockNode>(u)->board_[(row+len)*width_ + col] == -1) {
              vector<int> board(dynamic_pointer_cast<BlockNode>(u)->board_);
              board[pos] = -1; // move, empty
              board[new_pos] = id;
              board[(row+len)*width_ + col] = id;
              node_ptr new_node = make_shared<BlockNode>(board);
              new_node->parent_move_ = make_shared<BlockMove>(id, new_pos);
              insert_new_node(u, new_node, l);
            }
          }

          if (row - 1 >= 0) {
            int new_pos = (row - 1) * width_ + col; // move up
            if (dynamic_pointer_cast<BlockNode>(u)->board_[new_pos] == -1) {
              vector<int> board(dynamic_pointer_cast<BlockNode>(u)->board_);
              board[(row+len-1)*width_ + col] = -1; // move, empty
              board[new_pos] = id;
              board[(row-1+len-1)*width_ + col] = id;
              node_ptr new_node = make_shared<BlockNode>(board);
              new_node->parent_move_ = make_shared<BlockMove>(id, new_pos);
              insert_new_node(u, new_node, l);
            }
          }
        }
      }
    }

    size_t get_future_cost(const node_ptr& s)
    {
      const int pos = find(dynamic_pointer_cast<BlockNode>(s)->board_.begin(),
                           dynamic_pointer_cast<BlockNode>(s)->board_.end(),
                           target_id_) - dynamic_pointer_cast<BlockNode>(s)->board_.begin();
      if (pos == height_ * width_) return 0;
      else {
        const int len = block_length_[target_id_];
        const int row = pos / width_, col = pos % width_;
        size_t cost = (width_-1) - (col + len-1);
        // default target block usually set to move horizontally
        for (int i = 1; i < width_; ++i) { // test starts from 1
          int test_pos = (row * width_ + col) + len-1 + i;
          if (col + len-1 + i < width_) {
            cost += (-1 == dynamic_pointer_cast<BlockNode>(s)->board_[test_pos]) ? -1 : 1;
          }
          else break;
        }
        return cost;
      }
    }

    bool goal_test(const node_ptr& s)
    {
      // cannot directly judge by hash
      return s->future_cost_ == 0;
    }



    vector<int> block_id_;
    vector<int> block_length_;
    vector<int> block_move_direction_;
    int width_, height_;
    int target_id_;
};


int main()
{

  int id[] =      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int length[] =  {2, 3, 2, 3, 2, 3, 2, 2, 2, 2};
  // 1 <===> horizontally, 0 <===> vertically
  int direction[]={1, 0, 0, 1, 0, 0, 1, 1, 1, 1};
  int board[] = {1, -1, 2, 3, 3, 3,
                 1, -1, 2, 4,-1,-1,
                 1,  0, 0, 4,-1, 5,
                 6,  6, 7, 7,-1, 5,
                 8,  8, 9, 9,-1, 5,
                 -1,-1,-1,-1,-1,-1};

/*
  int id[] =      {0, 1, 2, 3, 4, 5, 6, 7};
  int length[] =  {2, 3, 3, 3, 2, 2, 2, 3};
  // 1 <===> horizontally, 0 <===> vertically
  int direction[]={1, 1, 0, 0, 0, 1, 0, 1};
  int board[] = {1, 1, 1, -1, -1, 2,
                -1,-1, 3, -1, -1, 2,
                0,  0, 3, -1, -1, 2,
                4, -1, 3, -1, 5,  5,
                4, -1,-1, -1, 6, -1,
                7,  7, 7, -1, 6, -1};
*/

  shared_ptr<Problem> p = make_shared<Block>("Block",
      6, 6,
      vector<int>(id, id+sizeof(id)/sizeof(id[0])),
      vector<int>(length, length+sizeof(length)/sizeof(length[0])),
      vector<int>(direction, direction+sizeof(direction)/sizeof(direction[0])),
      vector<int>(board, board+sizeof(board)/sizeof(board[0])),
      0);


  shared_ptr<Engine> e = make_shared<RBFS>();
  shared_ptr<node_list> m = make_shared<node_list>();
  e->search(p, m);
  e->show();
  for (auto& i : *m) i->show();
cout   << "|Closed|=" << e->closed_size_
               << ", max(open)=" << e->max_open_
               << ", depth=" << m->size()-1
               << ", max(depth)=" << e->max_depth_ << endl;
  return 0;
}
