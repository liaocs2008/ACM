namespace cpp RING

struct Node {
  1: required string host 
  2: required string succ
  3: required string pred
}

struct Message {
  1: required string from_host
  2: required i32    from_port

  3: required string to_host
  4: required i32    to_port 

  5: required string type
  6: list<string> content
}

service RING {
  void Register(1: string host, 2: i32 port)
  void Process(1: Message m)
}
