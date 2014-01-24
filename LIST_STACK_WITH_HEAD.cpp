#include <stdexcept>
#include <iostream>

using namespace std;

class Node{
public:
    Node(int v=0, Node* n=NULL, Node* p=NULL):
        value(v), next(n), prev(p)
    {}
    Node(const Node& other)
    {
        value = other.value;
        next = other.next;
        prev = other.prev;
    }
    Node& operator=(const Node& other)
    {
        value = other.value;
        next = other.next;
        prev = other.prev;
        return *this;
    }

    int value;
    Node* next;
    Node* prev;
};

class LIST_STACK_WITH_HEAD{
public:
    LIST_STACK_WITH_HEAD()
    {
        head = new Node();
        head->next = head;
        head->prev = head;
    }
    ~LIST_STACK_WITH_HEAD()
    {
        while(head->next!=head) remove(head->next);
    }
    Node* search(int v)
    {
        Node* ptr=head->next;
        while((ptr!=head) && (ptr->value!=v))
            ptr = ptr->next;
        return ptr;
    }
    void insert(Node* p)
    {
        p->next = head->next;
        p->prev = head;
        head->next->prev = p;
        head->next = p;
    }
    void remove(Node* p)
    {
        p->prev->next = p->next;
        p->next->prev = p->prev;
    }
    void print()
    {
        for (Node *p=head->next; p!=head; p=p->next)
            cout << p->value << '\n';
    }
private:
    Node* head;
};

int main()
{
    LIST_STACK_WITH_HEAD list;
    cout << "INSERT TEST\n";
    for (int i=0; i<10; ++i)
    {
        Node *p=new Node(i);
        list.insert(p);
    }
    list.print();
    cout << "SEARCH & REMOVE TEST\n";
    for (int i=0; i<10; i+=2)
    {
        Node *p=list.search(i);
        if (p) list.remove(p);
    }
    list.print();
    return 0;
}
