#include "base.hpp"

#include <vector>
#include <algorithm>
#include <iomanip>
#include <future>
#include <chrono>


#define N 3 // side length of the board

class PuzzleNode : public Node {
  public:
    PuzzleNode(const int& blank, const vector<int>& board) : Node()
    {
      blank_ = blank;
      board_.resize(board.size());
      copy(board.begin(), board.end(), board_.begin());

      // define hash function
      unique_hash_ = 0;
      for(auto& i : board_) {
        unique_hash_ ^= i + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);
      }
    }
    ~PuzzleNode() {}

    void show()
    {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) cout << std::setw(3) << board_[i*N + j] << ", ";
        cout << endl;
      }
      cout << "(depth=" << depth_  << ", g(u)=" << current_cost_ << ", h(u)=" << future_cost_ << ")" << endl;
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
      int p = 0; //  position of blank, by default 0 stands for blank

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

    // relate new node to its parent node
    inline void insert_new_node(const node_ptr&u, node_ptr& new_node, shared_ptr<node_set>& l)
    {
      new_node->parent_ = u;
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
      }

      if (col > 0) { // move left block right
        int new_blank = row * width_ + (col - 1);
        vector<int> board(dynamic_pointer_cast<PuzzleNode>(u)->board_.begin(),
            dynamic_pointer_cast<PuzzleNode>(u)->board_.end());
        board[new_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[old_blank];
        board[old_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[new_blank];

        node_ptr new_node = make_shared<PuzzleNode>(new_blank, board);
        insert_new_node(u, new_node, l);
      }

      if (row < height_ - 1) { // move below block up
        int new_blank = (row + 1) * width_ + col;
        vector<int> board(dynamic_pointer_cast<PuzzleNode>(u)->board_.begin(),
            dynamic_pointer_cast<PuzzleNode>(u)->board_.end());
        board[new_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[old_blank];
        board[old_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[new_blank];

        node_ptr new_node = make_shared<PuzzleNode>(new_blank, board);
        insert_new_node(u, new_node, l);
      }

      if (col < width_ - 1) { // move right block left
        int new_blank = row * width_ + (col + 1);
        vector<int> board(dynamic_pointer_cast<PuzzleNode>(u)->board_.begin(),
            dynamic_pointer_cast<PuzzleNode>(u)->board_.end());
        board[new_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[old_blank];
        board[old_blank] = dynamic_pointer_cast<PuzzleNode>(u)->board_[new_blank];

        node_ptr new_node = make_shared<PuzzleNode>(new_blank, board);
        insert_new_node(u, new_node, l);
      }
    }

    size_t get_future_cost(const node_ptr& s)
    {
      size_t cost = 0;
      /*
      // misplaced blocks
      for (size_t i = 0; i < dynamic_pointer_cast<PuzzleNode>(goal_)->board_.size(); ++i) {
        if (dynamic_pointer_cast<PuzzleNode>(goal_)->board_[i] !=
            dynamic_pointer_cast<PuzzleNode>(s)->board_[i])
          cost += 1;
      }
      */

      // manhattan distance
      for (size_t i = 0; i < dynamic_pointer_cast<PuzzleNode>(goal_)->board_.size(); ++i) {
        cost += abs(dynamic_pointer_cast<PuzzleNode>(goal_)->board_[i] -
                    dynamic_pointer_cast<PuzzleNode>(s)->board_[i]);
      }


      return cost;
    }

    int width_, height_;
};



int main()
{
  //int init[] = {8, 1, 3, 4, 0, 2, 7, 6, 5};
  //int goal[] = {1, 2, 3, 4, 5, 6, 7, 8, 0};
  int init[] = {8,6,7,2,5,4,3,0,1};
  int goal[] = {6,4,7,8,5,0,3,2,1};


  shared_ptr<Problem> p = make_shared<Puzzle>("9-puzzle",
      vector<int>(init, init+sizeof(init)/sizeof(init[0])),
      vector<int>(goal, goal+sizeof(goal)/sizeof(goal[0])),
      N, N);

  shared_ptr<Engine> e = make_shared<A_Star>();
  shared_ptr<node_list> m = make_shared<node_list>();

  std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

  auto fut = std::async(std::launch::async, bind(&Engine::search, e, p, m));;
  std::chrono::milliseconds span (1000); // set time limit

  std::future_status status = fut.wait_for(span);
  if (status == std::future_status::timeout) {
    std::cout << "timeout\n";
  } else if (status == std::future_status::ready) {
    std::cout << "solution found!\n";
    // print solution, from init to goal
    e->show();
    for (auto& i : *m) i->show();
  }

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
  cout << "time=" << ms.count() << "ms"
    << ", |Closed|=" << e->closed_size_
    << ", max(open)=" << e->max_open_
    << ", depth=" << (m->size() > 0 ? m->size()-1 : m->size())
    << ", max(depth)=" << e->max_depth_ << endl;
  exit(0); // kill the thread launched by async()
}
