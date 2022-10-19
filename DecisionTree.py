from __future__ import annotations
from Examples import *
import math
from FeatureExtractor import FeatureMap
from IO import *


#
# Decision tree classifier class.
#
# Constructs a binary classification decision tree from the given dataset. This decision tree
# is capable of handling discrete valued features using a variation of the ID3 algorithm.
#
class DecisionTree:

    #
    # Constructor
    #
    # Takes as input a full set of training examples in
    # the form of an Examples object. A FeatureMap extracts
    # the feature names so as to allow the tree to generate
    # explanations when needed.
    #
    def __init__(self, examples: Examples):
        examples.store_possible_values()
        self.feature_map = FeatureMap(examples.get_column_headers())
        self.root = self.fit(examples)

    #
    # Wrapper method for the build_tree() method. Fits this decision tree
    # to the given training examples.
    #
    def fit(self, examples: Examples) -> Node:
        return self.build_tree(examples, examples, examples.features())

    #
    # Builds the decision tree out of the given training examples.
    #
    def build_tree(self, examples: Examples, parent_examples: Examples, features: {int}) -> Node:
        # No examples have the selected combination of features.
        if examples.are_empty():
            return self.plurality_leaf(parent_examples)
        # All examples have the same classification.
        if examples.have_unanimous_classification():
            return self.unanimous_leaf(examples)
        # Multiple training examples have the same feature vectors.
        if len(features) == 0:
            return self.plurality_leaf(examples)
        # Distinguishable examples remain. Build decision node.
        next_feature = self.most_important_feature(examples, features)
        new_features = features - {next_feature}
        node = Node()
        node.feature = next_feature
        # Grow subtree from this decision node.
        for value in examples.possible_values(next_feature):
            subset = examples.filter_by_feature(next_feature, value)
            child = self.build_tree(subset, examples, new_features)
            child.value = value
            node.children.append(child)
        return node

    #
    # Predict the appropriate class label (True or False)
    # for the given feature vector.
    #
    def predict(self, feature_vector: list, node: Node = None) -> (str, float):
        # Initial call. First node to compare against is root.
        if node is None:
            return self.predict(feature_vector, self.root)
        # Classification is possible. Return classification.
        if node.is_leaf():
            return node.get_classification()
        # Given node is a decision node. Find appropriate child node.
        else:
            feature_index = node.get_feature()
            feature_value = feature_vector[feature_index]
            child_links = node.child_links()
            current_child = next(child_links)
            while not current_child.passes_test(feature_value):
                current_child = next(child_links)
            return self.predict(feature_vector, current_child)

    #
    #  Generates a propositional logic decision rule that explains the
    #  predicted classification for the input feature vector.
    #
    def explanation(self, feature_vector: list, node: Node = None, rule="") -> str:
        # Initial call. First node to compare against is root.
        if node is None:
            return self.explanation(feature_vector, self.root, "IF ")
        # Classification is possible. Return classification.
        if node.is_leaf():
            classification, probability = node.get_classification()
            return rule + "\n" + "THEN " + classification + " WITH " + str(int(probability * 100)) + "% PROBABILITY"
        # All decision nodes below the root are preceded by a prior conjunct.
        if not node == self.root:
            rule += " AND " + "\n"
        # Given node is a decision node. Find appropriate child node explanation.
        feature_index = node.get_feature()
        feature_value = feature_vector[feature_index]
        feature_name = self.feature_map.name_of_index(feature_index)
        rule += feature_name + ": "
        child_links = node.child_links()
        current_child = next(child_links)
        while not current_child.passes_test(feature_value):
            current_child = next(child_links)
        rule += current_child.value
        return self.explanation(feature_vector, current_child, rule)

    #
    # Generates the full set of decision rules for this tree.
    # Each decision rule is of the form of a conditional logic
    # statement with a series of conjuncts and a final
    # classification. Each decision rule also comes with an
    # associated probability.
    #
    def decision_rules(self, node=None, rule="IF ", rules=None) -> list[(str, float)]:
        if node is None:
            return self.decision_rules(self.root, rule, [])
        if node.is_leaf():
            classification, probability = node.get_classification()
            rule += " " + "THEN " + classification.upper()
            rules.append((rule, float(probability)))
            return rules
        if not node == self.root:
            rule += " AND "
        feature_name = self.feature_map.name_of_index(node.get_feature())
        rule += feature_name + ": "
        for child in node.children:
            new_rule = rule + child.get_value()
            rules += self.decision_rules(child, new_rule, [])
        return rules

    #
    # Static method to construct a tree from the
    # dataset, generate the full set of decision
    # rules and save these to a .csv file.
    #
    @staticmethod
    def save_decision_rules():
        io = IO()
        data_set = io.read_from_dataset()
        classifier = DecisionTree(data_set)
        rules = classifier.decision_rules()
        rule_file = open(os.getcwd() + "\\" + "decision_rules.csv", 'a', newline='')
        data_writer = csv.writer(rule_file)
        for rule in rules:
            data_writer.writerow(rule)
        rule_file.close()

    #
    # Returns a leaf node with the plurality classification of the given
    # example set.
    #
    def plurality_leaf(self, examples: Examples) -> Node:
        classification, probability = examples.plurality_classification()
        return self.create_leaf_node(classification, probability)

    #
    # Returns a leaf node with the unanimous classification of the given
    # example set.
    #
    def unanimous_leaf(self, examples: Examples) -> Node:
        return self.create_leaf_node(examples.peek().classification(), 1)

    #
    # Returns the most important attribute to split on. Uses
    # the information gain metric.
    #
    def most_important_feature(self, examples: Examples, features: {int}) -> int:
        return max(features, key=lambda feature: self.gain(examples, feature))

    #
    # Returns the information gained by splitting the
    # given examples on the selected feature.
    #
    def gain(self, examples: Examples, feature: int) -> float:
        return self.entropy(examples) - self.remainder(examples, feature)

    #
    # Returns the entropy of the given example set.
    #
    def entropy(self, examples: Examples) -> float:
        if examples.are_empty(): return 0
        p = examples.positive_proportion()
        n = examples.negative_proportion()
        log2_p = 0 if p == 0 else math.log2(p)
        log2_n = 0 if n == 0 else math.log2(n)
        return -p * log2_p - n * log2_n

    #
    # Returns the expected entropy remaining after testing the
    # given attribute.
    #
    def remainder(self, examples: Examples, feature: int) -> float:
        subsets = examples.subsets_by_feature(feature)
        return math.fsum(subset.size() / examples.size() * self.entropy(subset) for subset in subsets)

    #
    # Creates a leaf node with the given classification and associated probability.
    #
    def create_leaf_node(self, classification: str, classification_prob: float) -> Node:
        node = Node()
        node.classification = classification
        node.classification_prob = classification_prob
        return node

    #
    # Returns the total number of nodes in
    # this decision tree.
    #
    def size(self, node):
        if node is None:
            return 0
        subtree_size = 0
        for child in node.child_links():
            subtree_size += self.size(child)
        return subtree_size + 1

    #
    # Generates the confusion matrix for this
    # tree from the given dataset and test set.
    #
    @staticmethod
    def print_confusion_matrix():
        io = IO()
        data_set = io.read_from_dataset()
        test_set = io.read_from_test_set()
        class_index = len(test_set[0]) - 1
        classifier = DecisionTree(data_set)
        true_pos = false_pos = true_neg = false_neg = hits = total = 0
        for feature_vector in test_set:
            classification = classifier.predict(feature_vector)[0]
            if classification == "True" and feature_vector[class_index] == "True":
                true_pos += 1
                hits += 1
            elif classification == "True" and feature_vector[class_index] == "False":
                false_pos += 1
            elif classification == "False" and feature_vector[class_index] == "True":
                false_neg += 1
            else:
                true_neg += 1
                hits += 1
            total += 1
        print("True positives: " + str(true_pos / total) + " False positives: " + str(false_pos / total))
        print("True negatives: " + str(true_neg / total) + " False negatives: " + str(false_neg / total))
        print("Total hits: " + str(hits / total))


#
# Node class.
#
# A Node is a fundamental unit and building block for the
# DecisionTree class.Each node has a link to several children,
# and either a feature or a classification (according to the node type).
#
class Node:

    # Constructor.
    def __init__(self):
        self.children = []
        self.feature = None
        self.value = None
        self.classification = None
        self.classification_prob = None
        self.condition = None

    #
    # Returns true if this node is a leaf,
    # false otherwise.
    #
    def is_leaf(self) -> bool:
        return self.classification is not None

    #
    # Returns the classification that this node
    # makes. Returns None if this node is not a
    # leaf.
    #
    def get_classification(self) -> (str, float):
        return self.classification, self.classification_prob

    #
    # Returns the index of the
    # feature this node splits on.
    #
    def get_feature(self) -> int:
        return self.feature

    #
    # Returns the value of this node. Only
    # defined if this node splits on a discrete
    # attribute.
    #
    def get_value(self) -> any:
        return self.value

    #
    # Returns a list of the child links this
    # node has.
    #
    def child_links(self) -> iter:
        return iter(self.children)

    #
    # Returns true if the given feature value meets the
    # branching condition for this node.
    #
    def passes_test(self, value: any) -> bool:
        return self.value == value