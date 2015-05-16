#include <string>
#include <memory>
#include <list>
#include <set>
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

class Node {
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


    size_t unique_hash_;
    size_t depth_;
    size_t current_cost_;
    size_t future_cost_;

    size_t f; // for RBFS

    move_ptr parent_move_;
    node_ptr parent_;
};

/*
struct node_compare {
  public:
    bool operator()(const node_ptr& u, const node_ptr& v) const {
      //return u->compare(v);
      return u->unique_hash_ < v->unique_hash_;
    }
};
typedef set<node_ptr, node_compare> node_set;
*/


struct node_hash {
  public:
    size_t operator()(const node_ptr& obj) const
    {
      return obj->unique_hash_;
    }
};
typedef unordered_set<node_ptr, node_hash> node_set;



class Problem {
  public:
    Problem() {}
    Problem(const string& name, const node_ptr& init, const node_ptr& goal) :
        name_(name), init_(init), goal_(goal) {}
    virtual ~Problem() {}

    virtual void get_successor(const node_ptr& s, shared_ptr<node_set>& l) {cout<<"get_successor() in BASE"<<endl;}
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

        cout << "|open|=" << open->size() << endl; for (auto& i : *open) i->show(); cout <<endl;
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
            improve(s, u, open, closed);
          }
        }

        //cout << "open:" << endl; for (auto& i : *open) i->show();
        //cout << "closed:" << endl; for (auto& i : *closed) i->show();
      } while (false == open->empty());

      closed_size_ += closed->size();
    }

    size_t max_open_;
    size_t max_depth_;
    size_t closed_size_;
};

class DFS : public Engine {
  public:
    DFS(int depth_limit=300) : Engine() {depth_limit_ = depth_limit;}
    void show() {cout << "DFS" << endl;}
    void extract(node_ptr& u, const shared_ptr<node_list>& open)
    {
      u = open->front();
      open->pop_front();
    }
    void insert(const node_ptr& u, shared_ptr<node_list>& l)
    {
      l->push_front(u);
    }

    void improve(const node_ptr& s,
                 const node_ptr& u,
                 shared_ptr<node_list>& open,
                 const shared_ptr<node_list>& closed,
                 const problem_ptr& p=nullptr)
    {
      if (u->depth_ >= depth_limit_) return; // limit adding new nodes to open list

      Engine::improve(s, u, open, closed);
    }

    size_t depth_limit_;
};

class IDS : public DFS {
  public:
    IDS(int depth_limit_upper_bound = 150) : DFS()
    {
       depth_limit_upper_bound_ = depth_limit_upper_bound;
    }
    void show() {cout << "IDS (Iterative deepening search)" << endl;}

    void search(const problem_ptr& p, shared_ptr<node_list>& path)
    {
      for (depth_limit_ = 1; depth_limit_ < depth_limit_upper_bound_; ++depth_limit_) {
        cout << "depth_limit_=" << depth_limit_ << endl;
        DFS::search(p, path);
        if (false == path->empty()) return;
      }
    }

    size_t depth_limit_upper_bound_;
};

class BFS : public Engine {
  public:
    void show() {cout << "BFS" << endl;}
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
  // UCS is like dijkstra, based on BFS search
  public:
    UCS() : Engine() {}
    void show() {cout << "UCS" << endl;}
    void extract(node_ptr& u, const shared_ptr<node_list>& open)
    {
      u = open->front();
      open->pop_front();
    }
    void insert(const node_ptr& u, shared_ptr<node_list>& l)
    {
      l->push_back(u);
    }
    void improve(const node_ptr& v,
                 const node_ptr& u,
                 shared_ptr<node_list>& open,
                 const shared_ptr<node_list>& closed,
                 const problem_ptr& p=nullptr)
    {
      for (auto& o : (*open)) {
        if (v->unique_hash_ == o->unique_hash_) {
          if (u->current_cost_ + v->parent_move_->cost() < o->current_cost_) {
            o->parent_ = u;
            o->parent_move_ = v->parent_move_;
            o->depth_ = u->depth_ + 1;
            o->current_cost_ = u->current_cost_ + v->parent_move_->cost();
          }
          return;
        }
      }

      for (auto& c : (*closed)) {
        if (v->unique_hash_ == c->unique_hash_) return;
      }

      insert(v, open);
    }
};

class Greedy : public Engine {
  public:
    Greedy() : Engine() {}
    void show() {cout << "Greedy" << endl;}

    void extract(node_ptr& u, const shared_ptr<node_list>& open)
    {
      u = open->front();
      open->pop_front();
    }

    void insert(const node_ptr& u, shared_ptr<node_list>& l)
    {
      // maintain an ordered list by future_cost_
      for (auto i = l->begin(); i != l->end(); ++i) {
        if ((*i)->future_cost_ > u->future_cost_) {
          l->insert(i, u);
          return;
        }
      }
      // in case u_cost is the least one
      l->push_back(u);
    }
};

class A_Star : public UCS {
  public:
    A_Star () : UCS() {}
    void show() {cout << "A_Star" << endl;}

    void extract(node_ptr& u, const shared_ptr<node_list>& open)
    {
      u = open->front();
      open->pop_front();
    }

    void insert(const node_ptr& u, shared_ptr<node_list>& l)
    {
      // maintain an ordered list by current_cost_ + future_cost_
      size_t u_cost = u->current_cost_ + u->future_cost_;

      for (auto i = l->begin(); i != l->end(); ++i) {
        size_t cost = (*i)->current_cost_ + (*i)->future_cost_;
        if (cost > u_cost) {
          l->insert(i, u);
          return;
        }
      }
      // in case u_cost is the least one
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
    void show() {cout << "IDA_Star (Iterative deepening A star)" << endl;}

    void improve(const node_ptr& s,
                 const node_ptr& u,
                 shared_ptr<node_list>& open,
                 const shared_ptr<node_list>& closed,
                 const problem_ptr& p=nullptr)
    {
      if (u->current_cost_ + u->future_cost_ >= cost_limit_) return; // limit adding new nodes to open list

      A_Star::improve(s, u, open, closed);
    }

    void search(const problem_ptr& p, shared_ptr<node_list>& path)
    {
      for (cost_limit_ = 1; cost_limit_ < cost_upper_bound_; ++cost_limit_) {
        cout << "cost_limit_=" << cost_limit_ << endl;
        A_Star::search(p, path);
        if (false == path->empty()) return;
      }
    }

    size_t cost_limit_;
    size_t cost_upper_bound_;
};

class SMA_Star : public A_Star {
  public:
    SMA_Star(size_t limit_nodes=192) : limit_nodes_(limit_nodes) {}

    void show() {cout << "SMA_Star (Simplified Memory-bounded A Star)" << endl;}

    void insert(const node_ptr& u, shared_ptr<node_list>& l)
    {
      // [deep...shallow]
      // [low cost...high cost]
      size_t u_cost = u->current_cost_ + u->future_cost_;

      for (auto i = l->begin(); i != l->end(); ++i) {
        if (u->depth_ >= (*i)->depth_) {
          size_t cost = (*i)->current_cost_ + (*i)->future_cost_;
          if (cost > u_cost) {
            l->insert(i, u);
            // pop out the shallowest worst node
            if (l->size()+1 >= limit_nodes_) {
              l->pop_back();
            }
            return;
          }
        }
      }

      if (l->size()+1 < limit_nodes_)
        l->push_back(u);
    }

    size_t limit_nodes_;
};

class RBFS : public DFS {
  public:
    shared_ptr<node_list> closed;

    RBFS() : DFS() {closed = make_shared<node_list>();}
    void show() {cout << "RBFS" << endl;}

#define INF 1e10
    void search(const problem_ptr& p, shared_ptr<node_list>& path)
    {
      p->init_->f = p->init_->future_cost_ + p->init_->current_cost_;
      aux(p, p->init_, INF, path); // 1e10 is for default upperbound
      closed_size_ = closed->size();
    }

    void aux(const problem_ptr&p, node_ptr& u, const size_t& upper_bound, shared_ptr<node_list>& path)
    {
      u->show();
      max_depth_ = max(u->depth_, max_depth_);
      if (p->goal_test(u)) {
        get_path(u, path);
        return;
      }
      else closed->push_front(u);

      shared_ptr<node_set> successors = make_shared<node_set>();
      p->get_successor(u, successors);
      if (successors->empty()) {
        u->f = INF;
        return;
      }

      for (auto& s : (*successors)) {
        s->f = s->current_cost_ + s->future_cost_;
        if (s->f < u->f) s->f = u->f;
      }

      shared_ptr<node_list> l = make_shared<node_list>();

      for (auto& s : (*successors)) {

        bool duplicate = false;
        for (auto& i : (*closed)) {
          if (i->unique_hash_ == s->unique_hash_) {
            duplicate = true;
            break;
          }
        }
        if (duplicate) continue;


        bool inserted = false;
        for (auto i = l->begin(); i != l->end(); ++i) {
          if ((*i)->f >= s->f) {
            l->insert(i, s);
            //cout << s->f << endl;
            inserted = true;
            break;
          }
        }

        if (false == inserted) {
          l->push_back(s);
          //cout << s->f << endl;
        }
      }

      cout << "current layer" << endl;
      for (auto& s : (*l)) {
        cout << s->f << endl;
      }

      size_t new_bound = 0;
      while (true) {
        auto lowest = l->front();
        l->pop_front();

        if (lowest->f > upper_bound) {
          u->f = lowest->f;
          return;
        }

        if (false == l->empty()) {
          new_bound = min( (*(l->begin()))->f, upper_bound);
        }
        else {
          new_bound = min(lowest->f, upper_bound);
        }
        aux(p, lowest, new_bound, path);
        if (false == path->empty()) return;
        else {
          // resort because lowest's f value will be updated
            bool inserted = false;
            for (auto i = l->begin(); i != l->end(); ++i) {
              if ((*i)->f >= lowest->f) {
                l->insert(i, lowest);
                //iter_swap(i, l->begin());
                //cout << s->f << endl;
                inserted = true;
                break;
              }
            }

            if (false == inserted) {
              l->push_back(lowest);
              //iter_swap(l->end(), l->end());
              //cout << s->f << endl;
            }
        }
      }
    }
};
