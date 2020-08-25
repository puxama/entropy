import unittest
import entropy


class TestEntropy(unittest.TestCase):
    labels = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 1, 0, 0, 0, 1]
    categorical_var = ['f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
                       'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm',
                       'a', 'a', 'a', 'a', 'a']

    def test_get_entropy(self):
        result = entropy.get_entropy([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 2)
        self.assertEqual(round(result, 2), 0.88)

    def test__get_categorical_entropy(self):
        output = [('a', 0.97, 0.22),
                  ('f', 0.99, 0.39),
                  ('m', 0, 0.39)]

        results = entropy._get_categorical_entropy(self.categorical_var, self.labels, 2)
        self.assertEqual(results, output)

    def test_get_information_gain(self):
        information_gain = entropy._get_information_gain(self.categorical_var,
                                                        self.labels, 2)
        self.assertEqual(information_gain, 0.23)


if __name__ == '__main__':
    unittest.main()
