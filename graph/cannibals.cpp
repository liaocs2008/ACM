#include "base.hpp"

#include <vector>
#include <algorithm>


class CannibalMove : public Move {
  public:
    CannibalMove() { boat_[0] = boat_[1] = 1; }// start from side1
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
      unique_hash_ = 0;
      unique_hash_ ^= side1_[0] + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);
      unique_hash_ ^= side1_[1] + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);

      // consider which way boat is at
      if (dynamic_pointer_cast<CannibalMove>(this->parent_move_)->boat_[0] > 0) {
        unique_hash_ ^= 0 + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);
      } else {
        unique_hash_ ^= 1 + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);
      }
    }
    ~CannibalNode() {}

    void show()
    {
      cout << "(" << side1_[0] << "," << side1_[1] << ")" << ", hash=" << unique_hash_
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
      move_ptr empty_move = make_shared<CannibalMove>();
      init_ = make_shared<CannibalNode>(init, empty_move);
      goal_ = make_shared<CannibalNode>(goal, empty_move);
      name_ = name;

      // in this case, the goal test is defined in a different way
      init_->future_cost_ = get_future_cost(init_);
    }
    ~Cannibal() {}

    inline void insert_new_node(const node_ptr&u, node_ptr& new_node, shared_ptr<node_set>& l)
    {
      new_node->parent_ = u;
      //new_node->parent_move_ = make_shared<WaterMove>(dynamic_pointer_cast<WaterNode>(new_node)->bottles_);
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

    bool goal_test(const node_ptr& s)
    {
      // cannot directly judge by hash which involves boat side
      return s->future_cost_ == 0;
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

  shared_ptr<Problem> p = make_shared<Cannibal>("Cannibals", init, goal, 3, 3);

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
