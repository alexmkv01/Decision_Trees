from __future__ import annotations
import copy


#
# Examples class.
#
# Utility and wrapper class for Example class. Training examples are a list of examples.
# This class allows a full dataset to be passed between classes and methods with useful
# utility methods for data manipulation and decision tree classification.
class Examples:

    def __init__(self):
        self.list = []
        self.value_matrix = []
        self.column_headers = []

    #
    # Appends an example to this collection of examples.
    #
    def append(self, example):
        self.list.append(example)

    #
    # Sets the list of column headers (feature names).
    #
    def set_column_headers(self, column_headers: list):
        self.column_headers = column_headers

    #
    # Returns the list of column headers (feature names).
    #
    def get_column_headers(self):
        return self.column_headers

    #
    # Returns the first example in this
    # collection.
    #
    def peek(self) -> Example:
        return self.list[0]

    #
    # Returns the number of examples in this
    # collection.
    #
    def size(self) -> int:
        return len(self.list)

    #
    # Performs a copy of this collection of examples.
    #
    def copy(self) -> Examples:
        new_examples = Examples()
        new_examples.list = copy.deepcopy(self.list)
        new_examples.value_matrix = copy.deepcopy(self.value_matrix)
        return new_examples

    #
    # Returns a set of integers representing
    # the features in this collection.
    #
    def features(self):
        return set([feature for feature in range(self.peek().feature_count())])

    #
    # Returns an an Examples wrapper object containing only those examples
    # with the selected feature equal to the provided value. Result may be empty.
    #
    def filter_by_feature(self, feature, value) -> Examples:
        new_examples = self.copy()
        new_examples.list = list(filter(lambda e: e.get_feature(feature) == value, new_examples.list))
        return new_examples

    #
    # Returns subsets of this collection of examples according to
    # the possible values of the provided feature. Subsets may be
    # empty.
    #
    def subsets_by_feature(self, feature):
        possible_values = self.possible_values(feature)
        return [self.filter_by_feature(feature, value) for value in possible_values]

    #
    # Returns true if every example in this collection of examples
    # has the same classification. False otherwise.
    #
    def have_unanimous_classification(self):
        return all([example.classification() == self.peek().classification() for example in self.list])

    #
    # Returns the most common classification for all examples
    # in the given example set, along with the proportion of
    # examples that have this classification.
    #
    def plurality_classification(self) -> (bool, float):
        positive = sum([example.classification() == "True" for example in self.list])
        negative = len(self.list) - positive
        classification = "True" if positive >= negative else "False"
        majority = positive if positive >= negative else negative
        return classification, majority / len(self.list)

    #
    # Returns true if this collection of examples is
    # empty. False otherwise.
    #
    def are_empty(self):
        return len(self.list) == 0

    #
    # Returns the proportion of examples in this collection of examples that are
    # negative. Assumes nonempty example collection.
    #
    def negative_proportion(self):
        return sum([example.classification() == 'False' for example in self.list]) / len(self.list)

    #
    # Returns the proportion of examples in this collection of examples that are
    # positive. Assumes nonempty example collection.
    #
    def positive_proportion(self):
        return sum([example.classification() == 'True' for example in self.list]) / len(self.list)

    #
    # Stores all of the possible feature values
    # for each feature (as seen in the training
    # set).
    #
    def store_possible_values(self):
        self.value_matrix = []
        columns = self.peek().feature_count()
        rows = len(self.list)
        # Iterate column wise over examples
        for column in range(columns):
            values = set()
            for row in range(rows):
                example = self.list[row]
                feature_value = example.get_feature(column)
                values.add(feature_value)
            self.value_matrix.append(values)

    #
    # Returns all possible values for the
    # feature as seen in the training set.
    #
    def possible_values(self, feature):
        return self.value_matrix[feature]

    #
    # Prints the examples contained in this
    # collection.
    #
    def print(self):
        for example in self.list: example.print()


#
# Example class.
#
# Wrapper class for training data. A training example consists of a
# single feature vector and a classification.
#
class Example:

    #
    # Constructor.
    #
    def __init__(self, features: list):
        self.boolean_classification_str = features.pop()
        self.features = features

    #
    # Returns the feature vector
    # for this example.
    #
    def get_features(self) -> list:
        return self.features

    #
    # Returns the number of features.
    #
    def feature_count(self) -> int:
        return len(self.features)

    #
    # Returns the value of the specified
    # feature for this example.
    #
    def get_feature(self, feature) -> any:
        return self.features[feature]

    #
    # Returns the classification for this
    # example.
    #
    def classification(self) -> str:
        return self.boolean_classification_str

    #
    # Prints this training example.
    #
    def print(self):
        print(self.features + [self.boolean_classification_str])
