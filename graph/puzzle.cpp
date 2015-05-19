#include "base.hpp"

#include <vector>
#include <algorithm>

//#define DEBUG_PRINT

class PuzzleNode : public Node {
  public:
    PuzzleNode(const int& blank, const vector<int>& board) : Node()
    {
      blank_ = blank;
      board_.resize(board.size());
      copy(board.begin(), board.end(), board_.begin());

      unique_hash_ = 0;
      for(auto& i : board_) {
        unique_hash_ ^= i + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);
      }
    }
    ~PuzzleNode() {}

    void show()
    {
      copy(board_.begin(), board_.end(), std::ostream_iterator<int>(std::cout, " "));
      cout << ", " << depth_  << ", " << current_cost_ << ", " << future_cost_ << endl;
    }

    int blank_; // position for the blank
    vector<int> board_;
};

class PuzzleMove : public Move {
  public:
    PuzzleMove(int block=-1) { block_ = block; }
    ~PuzzleMove() {}
    int block_; // move this block to the blank
};

class Puzzle : public Problem {
  public:
    Puzzle(const string& name, const vector<int>& init, const vector<int>& goal, int width, int height)
    {
      int p = 0;

      p = find(init.begin(), init.end(), 0) - init.begin();
      init_ = make_shared<PuzzleNode>(p, init);
      init_->parent_move_ = make_shared<PuzzleMove>();

      p = find(goal.begin(), goal.end(), 0) - goal.begin();
      goal_ = make_shared<PuzzleNode>(p, goal);
      goal_->parent_move_ = make_shared<PuzzleMove>();

      width_ = width;
      height_ = height;
      name_ = name;
    }
    ~Puzzle() {}

    inline void insert_new_node(const node_ptr&u, node_ptr& new_node, shared_ptr<node_set>& l)
    {
      new_node->parent_ = u;
      //new_node->parent_move_ = make_shared<PuzzleMove>(new_blank);
      new_node->parent_move_ = make_shared<PuzzleMove>(dynamic_pointer_cast<PuzzleNode>(new_node)->blank_);
      new_node->depth_ = u->depth_ + 1;
      new_node->current_cost_ = u->current_cost_ + new_node->parent_move_->cost();
      new_node->future_cost_ = this->get_future_cost(new_node);
      l->insert(new_node);
    }

    void get_successor(const node_ptr& u, shared_ptr<node_set>& l)
    {
      assert(true == l->empty());
      int old_blank = dynamic_pointer_cast<PuzzleNode>(u)->blank_;
      int row = old_blank / width_, col = old_blank % width_; //(row, col) for blank

      if (row > 0) { // move above block down
        int new_blank = (row - 1) * width_ + col;
        vector<int> board(dynamic_pointer_cast<PuzzleNode>(u)->board_.begin(),
            dynamic_pointer_cast<PuzzleNode>(u)->board_.end());
        board[new_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[old_blank];
        board[old_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[new_blank];

        node_ptr new_node = make_shared<PuzzleNode>(new_blank, board);
        insert_new_node(u, new_node, l);
#ifdef DEBUG_PRINT
        cout<< "new_node "; new_node->show();
#endif
      }

      if (col > 0) { // move left block right
        int new_blank = row * width_ + (col - 1);
        vector<int> board(dynamic_pointer_cast<PuzzleNode>(u)->board_.begin(),
            dynamic_pointer_cast<PuzzleNode>(u)->board_.end());
        board[new_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[old_blank];
        board[old_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[new_blank];

        node_ptr new_node = make_shared<PuzzleNode>(new_blank, board);
        insert_new_node(u, new_node, l);

#ifdef DEBUG_PRINT
        cout<< "new_node "; new_node->show();
#endif
      }

      if (row < height_ - 1) { // move below block up
        int new_blank = (row + 1) * width_ + col;
        vector<int> board(dynamic_pointer_cast<PuzzleNode>(u)->board_.begin(),
            dynamic_pointer_cast<PuzzleNode>(u)->board_.end());
        board[new_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[old_blank];
        board[old_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[new_blank];

        node_ptr new_node = make_shared<PuzzleNode>(new_blank, board);
        insert_new_node(u, new_node, l);

#ifdef DEBUG_PRINT
        cout<< "new_node "; new_node->show();
#endif
      }

      if (col < width_ - 1) { // move right block left
        int new_blank = row * width_ + (col + 1);
        vector<int> board(dynamic_pointer_cast<PuzzleNode>(u)->board_.begin(),
            dynamic_pointer_cast<PuzzleNode>(u)->board_.end());
        board[new_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[old_blank];
        board[old_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[new_blank];

        node_ptr new_node = make_shared<PuzzleNode>(new_blank, board);
        insert_new_node(u, new_node, l);

#ifdef DEBUG_PRINT
        cout<< "new_node "; new_node->show();
#endif
      }
    }

    size_t get_future_cost(const node_ptr& s)
    {
      size_t cost = 0;
      for (size_t i = 0; i < dynamic_pointer_cast<PuzzleNode>(goal_)->board_.size(); ++i) {
        if (dynamic_pointer_cast<PuzzleNode>(goal_)->board_[i] !=
            dynamic_pointer_cast<PuzzleNode>(s)->board_[i])
          cost += 1;
      }
      return cost;
    }

    int width_, height_;
};



int main()
{
  int init[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  //int goal[] = {1, 2, 5, 3, 4, 8, 6, 0, 7};
  int goal[] = {1, 2, 5, 4, 0, 8, 3, 6, 7};

  shared_ptr<Problem> p = make_shared<Puzzle>("9-puzzle",
      vector<int>(init, init+sizeof(init)/sizeof(init[0])),
      vector<int>(goal, goal+sizeof(goal)/sizeof(goal[0])),
      3, 3);

  shared_ptr<Engine> e = make_shared<RBFS>();
  shared_ptr<node_list> m = make_shared<node_list>();
  e->search(p, m);
  e->show();
  for (auto& i : *m) i->show();

  cout << "|Closed|=" << e->closed_size_
    << ", max(open)=" << e->max_open_
    << ", depth=" << m->size()-1
    << ", max(depth)=" << e->max_depth_ << endl;
  return 0;
}
