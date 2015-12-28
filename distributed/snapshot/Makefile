All:  
	thrift --gen cpp disos.thrift
	g++ -DHAVE_NETINET_IN_H main.cpp gen-cpp/disos_types.cpp gen-cpp/disos_constants.cpp gen-cpp/RING.cpp -lthrift -lboost_system -lboost_thread -std=c++11 -lpthread 

