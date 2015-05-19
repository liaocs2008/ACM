#include "base.hpp"

#include <vector>
#include <algorithm>
#include <future>
#include <chrono>

class CannibalMove : public Move {
  public:
    CannibalMove(int boat[])
    {
      boat_[0] = boat[0];
      boat_[1] = boat[1];
    }
    ~CannibalMove() {}
    // boat can diretly be applied to side1_[2]
    // (+cannibals, +missionaries) for boat from side 2 to side 1
    // (-cannibals, -missionaries) for boat from side 1 to side 2
    int boat_[2];
};

class CannibalNode : public Node {
  public:
    CannibalNode(int side1[], const move_ptr& parent_move) : Node()
    {
      parent_move_ = parent_move;
      side1_[0] = side1[0];
      side1_[1] = side1[1];

      // It important to consider which way boat is at
      unique_hash_ = 0;
      unique_hash_ ^= side1_[0] + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);
      unique_hash_ ^= side1_[1] + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);

      // check which side boat is
      if (dynamic_pointer_cast<CannibalMove>(this->parent_move_)->boat_[0] > 0) {
        unique_hash_ ^= 0 + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);
      } else {
        unique_hash_ ^= 1 + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);
      }
    }
    ~CannibalNode() {}

    void show()
    {
      cout << "(" << side1_[0] << "," << side1_[1] << ")"
           << ", depth=" << depth_  << ", c_cost=" << current_cost_ << ", f_cost=" << future_cost_ << endl;
    }

    // (cannibals, missionaries) for side 1
    // no need to store side 2 which can be derived from total and side 1
    int side1_[2];
};

class Cannibal : public Problem {
  public:
    Cannibal(const string& name, int init[], int goal[], int cannibals, int missionaries)
      : cannibals_(cannibals), missionaries_(missionaries)
    {
      int init_side[] = {1,1}; // start with boat at side 1
      int goal_side[] = {-1, -1}; // end with boat at side 2
      init_ = make_shared<CannibalNode>(init, make_shared<CannibalMove>(init_side));
      goal_ = make_shared<CannibalNode>(goal, make_shared<CannibalMove>(goal_side));
      name_ = name;
    }
    ~Cannibal() {}

    // relate new node to its parent node
    inline void insert_new_node(const node_ptr&u, node_ptr& new_node, shared_ptr<node_set>& l)
    {
      new_node->parent_ = u;
      new_node->depth_ = u->depth_ + 1;
      new_node->current_cost_ = u->current_cost_ + new_node->parent_move_->cost();
      new_node->future_cost_ = this->get_future_cost(new_node);
      l->insert(new_node);
    }

    void get_successor(const node_ptr& u, shared_ptr<node_set>& l)
    {
      assert(true == l->empty());
      const int side1_cannibals = dynamic_pointer_cast<CannibalNode>(u)->side1_[0];
      const int side1_missionaries = dynamic_pointer_cast<CannibalNode>(u)->side1_[1];

      assert(side1_cannibals >= 0 && side1_cannibals <= cannibals_);
      assert(side1_missionaries >= 0 && side1_missionaries <= missionaries_);

      if (side1_missionaries > 0 && side1_cannibals > side1_missionaries) return;

      // there are only five possible cases on the boat (capacity is 2)
      // {x, y}, x for cannibals, y for missionaries
      int new_side1[2], boat[2];
      const int boat_side = dynamic_pointer_cast<CannibalMove>(u->parent_move_)->boat_[0] > 0 ? -1 : 1;
      const int valid_boat[5][2] = {{2, 0}, {1, 0}, {1, 1}, {0,  1}, {0,  2}};

      for (int i = 0; i < 5; ++i) {
        boat[0] = boat_side * valid_boat[i][0];
        boat[1] = boat_side * valid_boat[i][1];
        new_side1[0] = side1_cannibals + boat[0];
        new_side1[1] = side1_missionaries + boat[1];
        if (new_side1[0] < 0) continue;
        else if (new_side1[0] > cannibals_) continue;
        else if (new_side1[1] < 0) continue;
        else if (new_side1[1] > missionaries_) continue;
        else {
          // missionaries on side1 will be eaten
          if ((new_side1[0] > new_side1[1]) && (new_side1[1] != 0)) continue;
          // missionaries on side2 will be eaten
          if ((cannibals_ - new_side1[0] > missionaries_ - new_side1[1]) && (missionaries_ != new_side1[1])) continue;
          node_ptr new_node = make_shared<CannibalNode>(new_side1, make_shared<CannibalMove>(boat));
          insert_new_node(u, new_node, l);
        }
      }

    }

    size_t get_future_cost(const node_ptr& s)
    {
      size_t cost = 0;
      cost  = abs(dynamic_pointer_cast<CannibalNode>(goal_)->side1_[0] -
          dynamic_pointer_cast<CannibalNode>(s)->side1_[0])
        +
        abs(dynamic_pointer_cast<CannibalNode>(goal_)->side1_[1] -
            dynamic_pointer_cast<CannibalNode>(s)->side1_[1]);
      return cost;
    }

    int cannibals_, missionaries_;
};



int main()
{
  int init[] = {3, 3};
  int goal[] = {0, 0};

  shared_ptr<Problem> p = make_shared<Cannibal>("Cannibals", init, goal, init[0], init[1]);

  shared_ptr<Engine> e = make_shared<SMA_Star>();
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
