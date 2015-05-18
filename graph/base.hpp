#include <string>
#include <memory>
#include <list>
#include <unordered_set>
#include <assert.h>
#include <algorithm>
#include <iterator>
#include <iostream>
using namespace std;

class Move;
class Node;
class Problem;
class State;

typedef shared_ptr<Node> node_ptr;
typedef shared_ptr<Move> move_ptr;
typedef shared_ptr<Problem> problem_ptr;

typedef list<node_ptr> node_list;
typedef list<move_ptr> move_list;

class Move {
  public:
    Move() {}
    virtual ~Move() {}
    virtual int cost() { return 1; }
};

class Node : public std::enable_shared_from_this<Node> {
  public:
    Node()
    {
      parent_move_ = nullptr;
      parent_ = nullptr;
      depth_ = unique_hash_ = 0;
      current_cost_ = future_cost_ = 0;
    }
    virtual ~Node() {}
    virtual void show() = 0;

    bool operator == (const Node& rhs) {return this->unique_hash_ == rhs.unique_hash_;}

    size_t depth_;
    size_t current_cost_; // g(u)
    size_t future_cost_;  // h(u)

    // apply hash value to help duplicate node detection
    // user defines the computation for hash values
    size_t unique_hash_;

    // most algorithms represent f(u) by g(u) + h(u), to save memory cost
    // this "f" value is introduced later, for RBFS
    size_t f;

    // pointer to help record tranversal path
    move_ptr parent_move_;
    node_ptr parent_;
};


// follow STL definition for "unordered_set"
// this structure "node_hash" is mandantory
namespace std {
template <>
struct hash<Node> {
  public:
    size_t operator()(const node_ptr& obj) const
    {
      return obj->unique_hash_;
    }
};
}
typedef unordered_set<node_ptr> node_set;


class Problem {
  public:
    Problem() {}
    Problem(const string& name, const node_ptr& init, const node_ptr& goal) :
        name_(name), init_(init), goal_(goal) {}
    virtual ~Problem() {}

    // each problem must have its own way of generating successors
    virtual void get_successor(const node_ptr& s, shared_ptr<node_set>& l) = 0;

    // default settings for goal_test and calculate h(u)
    virtual bool goal_test(const node_ptr& s) { return s->unique_hash_ == goal_->unique_hash_; }
    virtual size_t get_future_cost(const node_ptr& s) {return 0;}

    string name_;
    node_ptr init_;
    node_ptr goal_;
};

class Engine {
  public:
    Engine() {max_open_ = max_depth_ = closed_size_ = 0;}
    virtual void show() = 0;
    virtual void extract(node_ptr& u, const shared_ptr<node_list>& open) = 0;
    virtual void insert(const node_ptr& u, shared_ptr<node_list>& closed) = 0;

    virtual void improve(const node_ptr& s,
                         const node_ptr& u,
                         shared_ptr<node_list>& open,
                         const shared_ptr<node_list>& closed,
                         const problem_ptr& p=nullptr)
    {
      for (auto& o : (*open)) {
        if (s->unique_hash_ == o->unique_hash_) return;
      }

      for (auto& c : (*closed)) {
        if (s->unique_hash_ == c->unique_hash_) return;
      }

      insert(s, open);
    }

    // return the path from initial node to goal node
    // path is a stack here, top for initial node, tail for goal node
    void get_path(const node_ptr& u, shared_ptr<node_list>& path)
    {
      assert(true == path->empty());
      node_ptr v = u;

      path->push_back(v);
      while(nullptr != v->parent_) {
        v = v->parent_;
        path->push_front(v);
      }
    }

    virtual void search(const problem_ptr& p, shared_ptr<node_list>& path)
    {
      shared_ptr<node_list> closed = make_shared<node_list>();
      shared_ptr<node_list> open = make_shared<node_list>();
      node_ptr u = nullptr;

      insert(p->init_, open);
      do {
        max_open_ = max(open->size(), max_open_);

        //cout << "|open|=" << open->size() << endl; for (auto& i : *open) i->show(); cout <<endl;
        extract(u, open); //cout << "extract: "; u->show();
        insert(u, closed);

        max_depth_ = max(u->depth_, max_depth_);

        if (true == p->goal_test(u)) {
          get_path(u, path);
          closed_size_ = closed->size();
          return;
        } else {
          shared_ptr<node_set> successors = make_shared<node_set>();
          p->get_successor(u, successors);
          for (auto& s : (*successors)) {
            improve(s, u, open, closed, p);
          }
        }

      } while (false == open->empty());

      // arrive here only if solution is not found
      closed_size_ += closed->size();
    }

    // statistics
    size_t max_open_;
    size_t max_depth_;
    size_t closed_size_;
};

class DFS : public Engine {
  public:
    DFS(int depth_limit=300) : Engine() {depth_limit_ = depth_limit;}
    void show() {cout << "DFS (Depth First Search, depth cut off " << depth_limit_ << ")" << endl;}

    // non-recursive DFS is basically maintaining a stack
    void extract(node_ptr& u, const shared_ptr<node_list>& open)
    {
      u = open->front();
      open->pop_front();
    }
    void insert(const node_ptr& u, shared_ptr<node_list>& l)
    {
      l->push_front(u);
    }

    // with depth limit cut off
    void improve(const node_ptr& s,
                 const node_ptr& u,
                 shared_ptr<node_list>& open,
                 const shared_ptr<node_list>& closed,
                 const problem_ptr& p=nullptr)
    {
      if (u->depth_ >= depth_limit_) return;
      else Engine::improve(s, u, open, closed, p);
    }

    size_t depth_limit_;
};

class IDS : public DFS {
  public:
    IDS(int depth_limit_upper_bound = 150) : DFS()
    {
       depth_limit_upper_bound_ = depth_limit_upper_bound;
    }
    void show() {cout << "IDS (Iterative Deepening Search)" << endl;}

    void search(const problem_ptr& p, shared_ptr<node_list>& path)
    {
      // try different depth limit for cutting off
      for (depth_limit_ = 1; depth_limit_ < depth_limit_upper_bound_; ++depth_limit_) {
        cout << "depth_limit_=" << depth_limit_ << endl;
        DFS::search(p, path);

        // check if solution is found
        if (false == path->empty()) return;
      }
    }

    size_t depth_limit_upper_bound_;
};

class BFS : public Engine {
  public:
    BFS() : Engine() {}
    void show() {cout << "BFS (Breadth First Search)" << endl;}

    // non-recursive BFS is basically maintaining a queue
    void extract(node_ptr& u, const shared_ptr<node_list>& open)
    {
      u = open->front();
      open->pop_front();
    }
    void insert(const node_ptr& u, shared_ptr<node_list>& l)
    {
      l->push_back(u);
    }
};

class UCS : public Engine {
  public:
    UCS() : Engine() {}
    void show() {cout << "UCS (Uniform Cost Search)" << endl;}

    // UCS is like dijkstra, maintaining a priority queue by g(u)
    void insert(const node_ptr& u, shared_ptr<node_list>& l)
    {
      // insertion sort to maintain the queue
      // the head is the node with the least g(u), i.e., cost so far
      for (auto i = l->begin(); i != l->end(); ++i) {
        if ((*i)->current_cost_ > u->current_cost_) {
          l->insert(i, u);
          return;
        }
      }
      // arrive here only if l is empty, or g(u) is the largest so far
      l->push_back(u);
    }
    void extract(node_ptr& u, const shared_ptr<node_list>& open)
    {
      u = open->front();
      open->pop_front();
    }

    void improve(const node_ptr& s,
                 const node_ptr& u,
                 shared_ptr<node_list>& open,
                 const shared_ptr<node_list>& closed,
                 const problem_ptr& p=nullptr)
    {
      for (auto o = open->begin(); o != open->end(); ++o) {
        if (s->unique_hash_ == (*o)->unique_hash_) {
          // successor node found in open list! ("s" == "o")
          // "s" is a successor node of "u"
          // "o" is the corresponding node of "s", already in open list
          if (u->current_cost_ + s->parent_move_->cost() < (*o)->current_cost_) {
            // a better path to node "o" (or "s") is found, update priority queue
            open->erase(o);
            insert(s, open);
          }
          return;
        }
      }

      for (auto& c : (*closed)) {
        if (s->unique_hash_ == c->unique_hash_) return;
      }

      insert(s, open);
    }
};

class Greedy : public Engine {
  public:
    Greedy() : Engine() {}
    void show() {cout << "GS (Greedy Search)" << endl;}

    void insert(const node_ptr& u, shared_ptr<node_list>& l)
    {
      // insertion sort to maintain the queue
      // the head is the node with the least h(u), i.e., cost to goal node
      for (auto i = l->begin(); i != l->end(); ++i) {
        if ((*i)->future_cost_ > u->future_cost_) {
          l->insert(i, u);
          return;
        }
      }
      // arrive here only if l is empty, or h(u) is the largest so far
      l->push_back(u);
    }
    void extract(node_ptr& u, const shared_ptr<node_list>& open)
    {
      u = open->front();
      open->pop_front();
    }
};

class A_Star : public UCS {
  public:
    // A* is identical to UCS except the way sorting its priority queue (from textbook)
    A_Star () : UCS() {}
    void show() {cout << "A*" << endl;}

    void extract(node_ptr& u, const shared_ptr<node_list>& open)
    {
      u = open->front();
      open->pop_front();
    }

    void insert(const node_ptr& u, shared_ptr<node_list>& l)
    {
      // insertion sort to maintain the queue
      // the head is the node with the least g(u)+h(u)
      size_t u_cost = u->current_cost_ + u->future_cost_;

      for (auto i = l->begin(); i != l->end(); ++i) {
        size_t cost = (*i)->current_cost_ + (*i)->future_cost_;
        if (cost > u_cost) {
          l->insert(i, u);
          return;
        }
      }
      // arrive here only if l is empty, or g(u)+h(u) is the largest so far
      l->push_back(u);
    }
};


class IDA_Star : public A_Star {
  public:
    IDA_Star(int cost_upper_bound = 60) : A_Star()
    {
       cost_limit_ = 1;
       cost_upper_bound_ = cost_upper_bound;
    }
    void show() {cout << "IDA* (Iterative deepening A*)" << endl;}

    // with cost limit cut off
    void improve(const node_ptr& s,
                 const node_ptr& u,
                 shared_ptr<node_list>& open,
                 const shared_ptr<node_list>& closed,
                 const problem_ptr& p=nullptr)
    {
      if (u->current_cost_ + u->future_cost_ >= cost_limit_) return;
      else A_Star::improve(s, u, open, closed, p);
    }

    // try different cost limit for cutting off
    void search(const problem_ptr& p, shared_ptr<node_list>& path)
    {
      for (cost_limit_ = 1; cost_limit_ < cost_upper_bound_; ++cost_limit_) {
        cout << "cost_limit_=" << cost_limit_ << endl;
        A_Star::search(p, path);

        // check if solution found
        if (false == path->empty()) return;
      }
    }

    size_t cost_limit_;
    size_t cost_upper_bound_;
};

class SMA_Star : public A_Star {
  public:
    // maintain a priority queue by its size
    // throw away the shallowest node with the largest cost
    SMA_Star(size_t limit_nodes=192) : A_Star() {limit_nodes_=limit_nodes;}
    void show() {cout << "SMA* (Simplified Memory-bounded A*)" << endl;}

    void insert(const node_ptr& u, shared_ptr<node_list>& l)
    {
      // [deep....................shallow]
      //   |  \______               |   \______
      //   |         \              |          \
      // [low cost...high cost]   [low cost...high cost]
      size_t u_cost = u->current_cost_ + u->future_cost_;

      for (auto i = l->begin(); i != l->end(); ++i) {
        if (u->depth_ >= (*i)->depth_) {
          size_t cost = (*i)->current_cost_ + (*i)->future_cost_;
          if (cost > u_cost) {
            l->insert(i, u);
            if (l->size()+1 >= limit_nodes_) {
              // pop out the shallowest worst node
              l->pop_back();
            }
            return;
          }
        }
      }

      // arrive here only if l is empty, or g(u)+h(u) is the largest so far
      if (l->size()+1 < limit_nodes_)
        l->push_back(u);
    }

    size_t limit_nodes_;
};

class RBFS : public DFS {
  public:
    shared_ptr<node_list> closed_;

    RBFS() : DFS() {closed_ = make_shared<node_list>();}
    void show() {cout << "RBFS (Recursive Best First Search)" << endl;}

    void insert(const node_ptr& u, shared_ptr<node_list>& l)
    {
      // insertion sort to maintain the queue
      // the head is the node with the least f
      for (auto i = l->begin(); i != l->end(); ++i) {
        if ((*i)->f >= u->f) {
          l->insert(i, u);
          return;
        }
      }
      // arrive here only if l is empty, or f is the largest so far
      l->push_back(u);
    }


#define INF 1e10  // 1e10, pre-defined maximum cost
    void search(const problem_ptr& p, shared_ptr<node_list>& path)
    {
      p->init_->f = p->init_->future_cost_ + p->init_->current_cost_;
      aux(p, p->init_, INF, path);
      closed_size_ = closed_->size();
    }

    void aux(const problem_ptr&p, node_ptr& u, const size_t& upper_bound, shared_ptr<node_list>& path)
    {
      cout << "visited : upperbound=" << upper_bound << ", "; u->show();
      max_depth_ = max(u->depth_, max_depth_);
      // check if solution found
      if ((false == path->empty()) || (true == p->goal_test(u))) {
        get_path(u, path);
        return;
      }
      else {
        bool duplicate = false;
        for (auto& i : *closed_) {
          if (i->unique_hash_ == u->unique_hash_) {
            duplicate = true;
            break;
          }
        }
        if (false == duplicate) closed_->push_front(u);
      }

      // generate successor and put into priority queue
      shared_ptr<node_set> successors = make_shared<node_set>();
      p->get_successor(u, successors);
      if (successors->empty()) {
        u->f = INF;
        return;
      }

      // each group of successor has its own priority queue
      shared_ptr<node_list> l = make_shared<node_list>();
      for (auto& s : (*successors)) {
        s->f = s->current_cost_ + s->future_cost_;

        bool duplicate = false;
        for (auto& tmp : (*l)) {
          if ((s != tmp) && (s->unique_hash_ == tmp->unique_hash_)) {
            duplicate = true;
            break;
          }
        }
        if (duplicate) continue;
        else insert(s, l);
      }

      successors->clear(); // save memory
      if (true == l->empty()) return;

      size_t new_bound = 0;
      while (true) {
        cout << "current layer:" << u->depth_ << endl;
        for (auto &t : *l) t->show();

        auto lowest = l->front();
        l->pop_front();

        // do not use ">=", because "=" case can result in infite loop
        if (lowest->f > upper_bound) {
          u->f = lowest->f;
          return;
        }

        // new_bound is usually determined by second lowest node and upper_bound
        if (false == l->empty()) {
          // after pop out the lowest, front is the second lowest
          new_bound = min(l->front()->f, upper_bound);
        }
        else {
          new_bound = min(lowest->f, upper_bound);
        }
        aux(p, lowest, new_bound, path);
        if (false == path->empty()) return; // solution found
        else {
          // resort because lowest's f value may be updated
          insert(lowest, l);
        }
      }
    }
};
