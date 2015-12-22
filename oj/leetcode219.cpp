//Contains Duplicate II 
// Given an array of integers and an integer k, find out whether there there are two distinct indices i and j in the array such that nums[i] = nums[j] and the difference between i and j is at most k.

class Solution {

  public:

    static bool compare(const pair <int,int>& obj1, const pair <int,int>& obj2) {

      return obj1.first > obj2.first;

    }



    bool containsNearbyDuplicate(vector<int>& nums, int k) {

      vector< pair <int,int> > foo;

      for (int i=0; i<nums.size(); ++i) {

        foo.push_back(make_pair(nums[i], i));

      }

      sort(foo.begin(), foo.end(), compare);

      for (int i=1; i<foo.size(); ++i) {

        if (foo[i].first == foo[i-1].first) {

          if (abs(foo[i].second - foo[i-1].second) <= k) return true;

        }

      }

      return false;

    }

};
