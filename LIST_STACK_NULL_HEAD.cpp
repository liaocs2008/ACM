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

class LIST_STACK_NULL_HEAD{
public:
    LIST_STACK_NULL_HEAD(){head = NULL;}
    ~LIST_STACK_NULL_HEAD()
    {
        while(head) remove(head);
    }
    Node* search(int v)
    {
        Node* ptr=head;
        while((ptr) && (ptr->value!=v))
            ptr = ptr->next;
        return ptr;
    }
    void insert(Node* p)
    {
        p->prev = 0;
        p->next = head;
        if (head) head->prev = p;
        head = p;
    }
    void remove(Node* p)
    {
        if (p->prev)
            p->prev->next = p->next;
        else
            head = p->next;
        if (p->next)
            p->next->prev = p->prev;
    }
    void print()
    {
        for (Node* p=head; p!=NULL; p=p->next)
            cout << p->value << '\n';
    }
private:
    Node* head;
};

int main()
{
    LIST_STACK_NULL_HEAD list;
    cout << "INSERT TEST\n";
    for (int i=0; i<10; ++i)
    {
        Node* p = new Node(i);
        list.insert(p);
    }
    list.print();
    cout << "REMOVE & SEARCH TEST\n";
    for (int i=0; i<10; i=i+2)
    {
        Node *p = list.search(i);
        if (p) list.remove(p);
    }
    list.print();
    return 0;
}
