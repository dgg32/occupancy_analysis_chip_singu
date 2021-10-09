import unittest
import neighbor_analysis

class TestNeighborAnalysis(unittest.TestCase):
    def compare_nested_list(self, a, b):
        for l, r in zip(sorted(a), sorted(b)):
            if set(l) != set(r):
                return False
        
        return True
                
    def test_merge_dnb_list(self):
        """
        test whether merge_dnb_list can merge two lists with shared element
        """
        res = self.compare_nested_list(neighbor_analysis.merge_dnb_lists([['a','b','c'],['b','e','f'],['x','y','z']]), [['a','b','c','e','f'],['x','y','z']])
        self.assertTrue(res)

    def test_calculate_significance(self):
        """
        test whether calculate_significance runs correctly
        """

        # not significant
        res = neighbor_analysis.calculate_significance(1, 100, 0.98, 100)
        self.assertEqual(res, "")

        # signifcant
        res = neighbor_analysis.calculate_significance(1, 100, 0.80, 100)
        self.assertEqual(res, " (!!!)")

        # signifcant
        res = neighbor_analysis.calculate_significance(1, 100, 0.90, 100)
        self.assertEqual(res, " (!!)")

if __name__ == '__main__':
    unittest.main()