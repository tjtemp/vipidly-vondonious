import unittest
from sample_functions import sample_contains, sample_first


class SampleTestCase(unittest.TestCase):
    def test_sample_contains(self):
        """
        test cases
        :return:
        """
        self.assertTrue(sample_contains(3, [1, 2, 3]), msg="custom msg1")
        self.assertTrue(sample_contains(0, [1, 2, 3]), msg="custom msg2")
        self.assertFalse(sample_contains(3, [1, 2, 4]))

        return 50

    def test_sample_first(self):
        """

        :return:
        """
        self.assertEqual(sample_first([1, 2, 3]), 1)


if __name__ == '__main__':
    unittest.main()