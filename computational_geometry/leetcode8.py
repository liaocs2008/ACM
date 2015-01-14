class Solution:
  # @param digits, a list of integer digits
  # @return a list of integer digits
  def plusOne(self, digits):
    return [int(x) for x in str(int(''.join(str(d) for d in digits)) + 1)]

if __name__ == "__main__":
  s = Solution()
  print s.plusOne([1,2,34,56]) # they didn't check this test. Interesting.
