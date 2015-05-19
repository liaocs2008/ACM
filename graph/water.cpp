#include "base.hpp"

#include <vector>
#include <algorithm>
#include <future>
#include <chrono>

class WaterNode : public Node {
  public:
    WaterNode(int bottles[]) : Node()
    {
      bottles_[0] = bottles[0];
      bottles_[1] = bottles[1];

      // define hash computation
      unique_hash_ = 0;
      unique_hash_ ^= bottles[0] + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);
      unique_hash_ ^= bottles[1] + 0x9e3779b9 + (unique_hash_ << 6) + (unique_hash_ >> 2);
    }
    ~WaterNode() {}

    void show()
    {
      cout << "(" << bottles_[0] << "," << bottles_[1] << ")"
           << ", depth=" << depth_  << ", g=" << current_cost_ << ", h=" << future_cost_ << endl;
    }

    int bottles_[2];
};

class WaterMove : public Move {
  public:
    WaterMove() { pour_[0] = pour_[1] = 0; }
    WaterMove(int pour[]) { pour_[0] = pour[0]; pour_[1] = pour[1]; }
    ~WaterMove() {}
    int pour_[2];
};

class Water : public Problem {
  public:
    Water(const string& name, int volumn[], int bottles_init[], int bottles_target[])
    {
      volumn_[0] = volumn[0];
      volumn_[1] = volumn[1];

      init_ = make_shared<WaterNode>(bottles_init);
      init_->parent_move_ = make_shared<WaterMove>();

      goal_ = make_shared<WaterNode>(bottles_target);
      goal_->parent_move_ = make_shared<WaterMove>();

      name_ = name;
    }
    ~Water() {}

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
      const int bottle0 = dynamic_pointer_cast<WaterNode>(u)->bottles_[0];
      const int bottle1 = dynamic_pointer_cast<WaterNode>(u)->bottles_[1];
      int bottles[2], pour[2];

      // clear bottle0
      {
        pour[0] = bottle0;
        pour[1] = 0;
        bottles[0] = bottle0 - pour[0];
        bottles[1] = bottle1 - pour[1];
        node_ptr new_node = make_shared<WaterNode>(bottles);
        new_node->parent_move_ = make_shared<WaterMove>(pour);
        insert_new_node(u, new_node, l);
      }

      // clear bottle1
      {
        pour[0] = 0;
        pour[1] = bottle1;
        bottles[0] = bottle0 - pour[0];
        bottles[1] = bottle1 - pour[1];
        node_ptr new_node = make_shared<WaterNode>(bottles);
        new_node->parent_move_ = make_shared<WaterMove>(pour);
        insert_new_node(u, new_node, l);
      }

      // fill bottle0
      {
        pour[0] = bottle0 - volumn_[0];
        pour[1] = 0;
        bottles[0] = bottle0 - pour[0];
        bottles[1] = bottle1 - pour[1];
        node_ptr new_node = make_shared<WaterNode>(bottles);
        new_node->parent_move_ = make_shared<WaterMove>(pour);
        insert_new_node(u, new_node, l);
      }

      // fill bottle1
      {
        pour[0] = 0;
        pour[1] = bottle1 - volumn_[1];
        bottles[0] = bottle0 - pour[0];
        bottles[1] = bottle1 - pour[1];
        node_ptr new_node = make_shared<WaterNode>(bottles);
        new_node->parent_move_ = make_shared<WaterMove>(pour);
        insert_new_node(u, new_node, l);
      }

      // pour bottle0 to bottle1
      {
        pour[0] = (bottle0 <= volumn_[1]-bottle1) ? bottle0 : volumn_[1]-bottle1;
        pour[1] = 0;
        bottles[0] = bottle0 - pour[0];
        bottles[1] = bottle1 + pour[0];
        node_ptr new_node = make_shared<WaterNode>(bottles);
        new_node->parent_move_ = make_shared<WaterMove>(pour);
        insert_new_node(u, new_node, l);
      }

      // pour bottle1 to bottle0
      {
        pour[0] = 0;
        pour[1] = (bottle1 <= volumn_[0]-bottle0) ? bottle1 : volumn_[0]-bottle0;
        bottles[0] = bottle0 + pour[1];
        bottles[1] = bottle1 - pour[1];
        node_ptr new_node = make_shared<WaterNode>(bottles);
        new_node->parent_move_ = make_shared<WaterMove>(pour);
        insert_new_node(u, new_node, l);
      }

    }

    size_t get_future_cost(const node_ptr& s)
    {
      size_t cost = 0;
      cost  = abs(dynamic_pointer_cast<WaterNode>(goal_)->bottles_[0] -
                  dynamic_pointer_cast<WaterNode>(s)->bottles_[0])
              +
              abs(dynamic_pointer_cast<WaterNode>(goal_)->bottles_[1] -
                  dynamic_pointer_cast<WaterNode>(s)->bottles_[1]);
      return cost;
    }

    int volumn_[2];
};

int main()
{
  int vol[] = {100, 3};
  int init[] = {0, 0};
  int goal[] = {77, 0};

  shared_ptr<Problem> p = make_shared<Water>("Waters", vol, init, goal);

  shared_ptr<Engine> e = make_shared<DFS>();
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
