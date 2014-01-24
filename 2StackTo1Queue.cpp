#include <stdexcept>
#include <iostream>
#include <stack>

using namespace std;

class MYQUEUE{
public:
    void enqueue(int x)
    {
        S1.push(x);
    }
    void dequeue()
    {
        if (S2.empty()) load(S1, S2);
        if (S2.empty()) throw runtime_error("Empty MYQUEUE doesn't pop.");
        S2.pop();
    }
    int front()
    {
        if (S2.empty()) load(S1, S2);
        if (S2.empty()) throw runtime_error("Empty MYQUEUE doesn't have front element.");
        return S2.top();
    }
    int back()
    {
        if (S1.empty()) load(S2, S1);
        if (S1.empty()) throw runtime_error("Empty MYQUEUE doesn't have back element.");
        return S1.top();
    }
private:
    stack<int> S1;
    stack<int> S2;

    void load(stack<int>& src, stack<int>& dst)
    {
        while(!src.empty())
        {
            dst.push(src.top());
            src.pop();
        }
    }
};

int main()
{
    MYQUEUE q;
    cout << "ENQUEUE TEST\n"
            "usually you will see front element stays the same,\n"
            "while tail element update.\n";
    for (int i=0; i<10; ++i)
    {
        q.enqueue(i);
        cout << i << "th: " << q.front() << q.back() << '\n';
    }
    cout << "DEQUEUE TEST\n"
            "usually you will see tail element stays the same,\n"
            "while front element update.\n";
    for (int i=0; i<10; ++i)
    {
        cout << i << "th: " << q.front() << q.back() << '\n';
        q.dequeue();
    }
    return 0;
}
