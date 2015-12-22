// Contains Duplicate III 
// Given an array of integers, find out whether there are two distinct indices i and j in the array such that the difference between nums[i] and nums[j] is at most t and the difference between i and j is at most k.

class Solution {

  public:

    static bool compare(const pair <double,int>& obj1, const pair <double,int>& obj2) {

      return obj1.first > obj2.first;

    }



    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {

      vector< pair <double,int> > foo;

      for (int i=0; i<nums.size(); ++i) {

        foo.push_back(make_pair(nums[i], i));

      }

      sort(foo.begin(), foo.end(), compare);

      for (int i=1; i<foo.size(); ++i) {

        if (abs(foo[i].first - foo[i-1].first) <= t) {

          if (abs(foo[i].second - foo[i-1].second) <= k) return true;

        }

      }

      return false;

    }

};
