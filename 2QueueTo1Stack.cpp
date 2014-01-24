#include <stdexcept>
#include <iostream>
#include <queue>

using namespace std;

class MYSTACK{
public:
    void push(int x)
    {
        if (Q1.size() > 0) Q1.push(x);
        else Q2.push(x);
    }
    void pop()
    {
        if (Q1.size() > 0)
        {
            load(Q1, Q2);
            Q1.pop();
        }
        else
        {
            load(Q2, Q1);
            Q2.pop();
        }
    }
    int top()
    {
        if (Q1.size() > 0)
            return Q1.back();
        else
            return Q2.back();
    }
private:
    queue<int> Q1;
    queue<int> Q2;

    void load(queue<int>& src, queue<int>& dst)
    {
        while(src.size() > 1)
        {
            dst.push(src.front());
            src.pop();
        }
    }
};

int main()
{
    MYSTACK S;
    cout << "PUSH TEST\n";
    for (int i=0; i<10; ++i)
    {
        S.push(i);
        cout << i << "th: " << S.top() << '\n';
    }
    cout << "POP TEST\n";
    for (int i=0; i<10; ++i)
    {
        cout << i << "th: " << S.top() << '\n';
        S.pop();
    }
    return 0;
}
