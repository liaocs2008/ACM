#include "gen-cpp/RING.h"
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

#include <thrift/transport/TSocket.h>
#include <boost/thread/thread.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <thrift/server/TThreadedServer.h>

#include <cstdlib>
#include <ctime>
#include <set>
#include <unordered_map>
#include <fstream>

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;

using boost::shared_ptr;

using namespace  ::RING;

#define SNAPSHOT "Snapshot"
#define INVITE   "Invite"
#define PERMIT   "Permit"
#define JOIN     "Join"
#define TOKEN    "Token"
#define TOKEN_TIMEOUT  "Token_Timeout"
#define EVALUATE "Evaluate" // represent end of current snapshot

namespace std {

  // http://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key
  typedef pair<string, int> Key1; 

  template <>
    struct hash<Key1>
    {
      size_t operator()(const Key1& k) const
      {
        return hash<string>()(k.first) ^ (hash<int>()(k.second) << 1);
      }
    };

  typedef pair<Key1, Key1> Key2; 

  template <>
    struct hash<Key2>
    {
      size_t operator()(const Key2& k) const
      {
        return hash<Key1>()(k.first) ^ (hash<Key1>()(k.second) << 1);
      }
    };

}


std::set< std::pair<std::string, int> > complete_list, peers;
std::vector< std::pair<std::string, int> > pending_list;
bool snap_flag = false;
bool token_flag = false;
uint32_t counter = 0; // for determining timeout
Node state;

boost::mutex mtx_, io_mtx_, snp_mtx_;



void print_message(const Message& m) {
  boost::mutex::scoped_lock lock(io_mtx_);

  std::cout << "[ FROM " 
    << m.from_host << ":" << m.from_port << " TO "
    << m.to_host << ":" << m.to_port << ", "
    << m.type << ", ";
  for (auto it = m.content.begin(); it != m.content.end(); ++it) {
    std::cout << it->c_str() << ", ";
  }
  std::cout << " ]" << std::endl;
}

class Client {
  public:
    Client(std::string host_addr, int port) {
      boost::shared_ptr<TSocket> socket(new TSocket(host_addr, port));
      boost::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
      boost::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
      client_ = new RINGClient(protocol);
      transport->open();

      host_ = host_addr;
      port_ = port;
    }

    ~Client() {
      if (client_) delete client_;
    }

    void Register(const std::string& host, const int32_t port) {
      client_->Register(host, port);

      boost::mutex::scoped_lock scoped_lock(mtx_);
      std::cout << "REGISTER " 
        << host  << ":" << port << " AT " 
        << host_ << ":" << port_ << std::endl;
    }

    void Process(const Message& m) {
      client_->Process(m);
      //print_message(m);
    }

  private:
    std::string host_;
    int port_;
    RINGClient *client_ = NULL;
};


class RINGHandler : virtual public RINGIf {
  public:
    RINGHandler() {
    }

    void Update(const Message& m) { // this is for automata of ring formation

      Message response;
      response.from_host = m.to_host;
      response.from_port = m.to_port;
      response.to_host = m.from_host;
      response.to_port = m.from_port;

      if (m.type == INVITE) {
        if (state.pred.empty() && state.succ.empty()) {
          response.type = JOIN;
          Client c(response.to_host, response.to_port);
          c.Process(response);
        }
      } else if (m.type == PERMIT && !token_flag) {
        boost::mutex::scoped_lock scoped_lock(mtx_);

        state.pred = m.from_host + ":" + std::to_string(m.from_port);
        state.succ = m.content[1];
        token_flag = true;

        boost::mutex::scoped_lock lock(io_mtx_);
        std::cout << "Token Got, " 
                  << "succ=" << state.succ << ", "
                  << "pred=" << state.pred << std::endl;
      } else if (m.type == JOIN && token_flag) {
        boost::mutex::scoped_lock scoped_lock(mtx_);
        pending_list.push_back({m.from_host, m.from_port});

        //response.type = PERMIT;
        //response.content.push_back(TOKEN);
        //response.content.push_back(state.succ);
        //Client c(response.to_host, response.to_port);
        //c.Process(response);
        //state.succ = m.from_host + ":" + std::to_string(m.from_port);
        //token_flag = false;
      } else if (m.type == TOKEN_TIMEOUT && !token_flag) {
        boost::mutex::scoped_lock scoped_lock(mtx_);
        token_flag = true;

        boost::mutex::scoped_lock lock(io_mtx_);
        std::cout << "Token Got" << std::endl;
      }

    }

    void Register(const std::string& host, const int32_t port) {
      std::cout << host << ":" << port << " REGISTERED" << std::endl;
      boost::mutex::scoped_lock lock(snp_mtx_);
      peers.insert({host, port});
    }

    void Process(const Message& m) {

      print_message(m);

      std::pair<std::string, int> from, to;
      to.first = m.to_host;
      to.second = m.to_port;
      from.first = m.from_host;
      from.second = m.from_port;


      if (m.type != SNAPSHOT) {
        Update(m); // go to ring-formation layer
        if (snap_flag == true) { // add to log
          print_message(m);
        }
      }
      else {
        boost::mutex::scoped_lock lock(snp_mtx_);
        complete_list.insert(from);
        if (snap_flag == false) {
          snap_flag = true;
          for (auto it = peers.begin(); it != peers.end(); ++it) {
            if (it->first == m.from_host) continue;
            Client c(it->first, it->second);
            c.Process(m);
          }
        }
        else {
          if (complete_list == peers) {
            Message tmp = m;
            tmp.type = EVALUATE;
            print_message(tmp);

            complete_list.clear();
            snap_flag = false;
          }
        }
      }
    }

  private:
};


void server_thread(int port) {
  shared_ptr<RINGHandler> handler(new RINGHandler());
  shared_ptr<TProcessor> processor(new RINGProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(port));
  shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());

  TThreadedServer server(processor, serverTransport, transportFactory, protocolFactory);
  server.serve();
}





void client_thread(
    const std::set< std::pair<std::string, int> >& s, 
    const std::pair<std::string, int>& local 
    ) {

  std::unordered_map< std::pair<std::string, int> , shared_ptr<Client> > peers_pool;

  boost::this_thread::sleep(boost::posix_time::seconds(6)); // wait for others

  for (auto it = s.begin(); it != s.end(); ++it) {
    shared_ptr<Client> p(new Client(it->first, it->second));
    p->Register(local.first, local.second);
    peers_pool.insert({std::make_pair(it->first, it->second), p});
  }
  std::cout << "Connections are all eastablished!" << std::endl;

  boost::this_thread::sleep(boost::posix_time::seconds(6)); // wait for others

  if (local.second == 10006) { // this is observer

    while (true) {
      Message m; 
      m.type = SNAPSHOT;
      m.from_host = local.first;
      m.from_port = local.second;
      while (true) {
        // broadcast snapshot message
        for (auto it = peers_pool.begin(); it != peers_pool.end(); ++it) {
          m.to_host = it->first.first;
          m.to_port = it->first.second;
          it->second->Process(m);
        }

        boost::this_thread::sleep(boost::posix_time::seconds(1));
      }
    }

  } else if (local.second == 10001) { // this process start inviting others
    Message m; 
    m.type = INVITE;
    m.from_host = local.first;
    m.from_port = local.second;

    {
      boost::mutex::scoped_lock scoped_lock(mtx_);
      state.pred = local.first + ":" + std::to_string(local.second);
      state.succ = state.pred;
      token_flag = true;
    }
    for (auto it = peers_pool.begin(); it != peers_pool.end(); ++it) {
      m.to_host = it->first.first;
      m.to_port = it->first.second;
      it->second->Process(m);
      {
        boost::mutex::scoped_lock lock(io_mtx_);
        std::cout << "Finish inviting " << m.to_host << ":" << m.to_port << std::endl;
      }
    }
  }

  srand (time(NULL)); // for pick process in pending_list

  while (true) {

    if (token_flag) {
      Message m; 
      m.from_host = local.first;
      m.from_port = local.second;

      if (!pending_list.empty()) {
        boost::mutex::scoped_lock scoped_lock(mtx_);
        m.type = PERMIT;
        int x = rand() % pending_list.size();
        m.to_host = pending_list[x].first;
        m.to_port = pending_list[x].second;
        m.content.push_back(TOKEN);
        m.content.push_back(state.succ);
        Client c(m.to_host, m.to_port);
        c.Process(m);

        state.succ = m.to_host + ":" + std::to_string(m.to_port);
        token_flag = false;
        pending_list.clear();
        counter = 0;
      }
      else {
        //{
        //  boost::mutex::scoped_lock lock(io_mtx_);
        //  std::cout << "keep inviting" << std::endl;
        //}

        if (counter < 3) {
          for (auto it = peers_pool.begin(); it != peers_pool.end(); ++it) {
            m.type = INVITE;
            m.to_host = it->first.first;
            m.to_port = it->first.second;
            it->second->Process(m);
            //{
            //  boost::mutex::scoped_lock lock(io_mtx_);
            //  std::cout << "Finish inviting " << m.to_host << ":" << m.to_port << std::endl;
            //}
          }
        } else { // timeout for holding token
          m.type = TOKEN_TIMEOUT;

          int colon = state.succ.find(':');
          m.to_host = state.succ.substr(0, colon);
          m.to_port = std::stoi(state.succ.substr(colon+1));

          Client c(m.to_host, m.to_port);
          c.Process(m);
          token_flag = false;
          counter = 0;
        }

        counter++;
      }

      boost::this_thread::sleep(boost::posix_time::seconds(3));

    } 
  }
}



int main(int argc, char *argv[]) {

  // configurations
  //
  std::ifstream infile;
  infile.open(argv[1], std::ifstream::in);
  assert(infile.is_open());

  std::pair<std::string, int> local;
  infile >> local.first >> local.second;

  int N = 0;
  infile >> N;
  std::set< std::pair<std::string, int> > peers;
  for (int i = 0; i < N; ++i)
  {
    std::string host;
    int port;
    infile >> host >> port;
    peers.insert( make_pair(host, port) );
  }
  infile.close();


  // start algorithm
  //
  boost::thread server(server_thread, local.second);
  boost::thread client(client_thread, peers, local);
  server.join();
  client.join();

  return 0;
}
