import math

class Node:
    def __init__(self, attribute=None, branches=None, classification=None):
        self.attribute = attribute
        self.branches = branches
        self.classification = classification

def aq11(examples, attributes, default):
    # Check for empty examples
    if not examples:
        return Node(classification=default)
    
    # Check if all examples have the same classification
    classification = examples[0][-1]
    if all(example[-1] == classification for example in examples):
        return Node(classification=classification)
    
    # Check for empty attributes
    if not attributes:
        return Node(classification=most_common_classification(examples))
    
    # Choose the best attribute to split on
    best_attribute = choose_attribute(examples, attributes)
    
    # Split examples on best_attribute
    branches = {}
    for value in set(example[best_attribute] for example in examples):
        new_examples = [example for example in examples if example[best_attribute] == value]
        new_attributes = [attr for attr in attributes if attr != best_attribute]
        branches[value] = aq11(new_examples, new_attributes, most_common_classification(examples))
    
    return Node(attribute=best_attribute, branches=branches)

def choose_attribute(examples, attributes):
    # Compute the information gain for each attribute
    info_gain = {}
    for attribute in attributes:
        info_gain[attribute] = compute_information_gain(examples, attribute)
    
    # Return the attribute with the highest information gain
    return max(info_gain, key=info_gain.get)

def compute_information_gain(examples, attribute):
    # Compute the entropy of the examples
    entropy_s = compute_entropy([example[-1] for example in examples])
    
    # Compute the weighted average of the entropies of each branch
    values = set(example[attribute] for example in examples)
    entropy_s_v = 0
    for value in values:
        subset = [example for example in examples if example[attribute] == value]
        entropy_s_v += len(subset) / len(examples) * compute_entropy([example[-1] for example in subset])
    
    # Compute the information gain
    return entropy_s - entropy_s_v

def compute_entropy(labels):
    # Count the number of occurrences of each label
    label_counts = {}
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    # Compute the entropy
    entropy = 0
    for count in label_counts.values():
        p = count / len(labels)
        entropy -= p * math.log2(p)
    
    return entropy

def most_common_classification(examples):
    # Count the number of occurrences of each classification
    classification_counts = {}
    for example in examples:
        classification = example[-1]
        if classification in classification_counts:
            classification_counts[classification] += 1
        else:
            classification_counts[classification] = 1
    
    # Return the classification with the highest count
    return max(classification_counts, key=classification_counts.get)

def classify(example, tree):
    if tree.classification is not None:
        return tree.classification
    
    value = example[tree.attribute]
    if value not in tree.branches:
        return most_common_classification([example])
    
    branch = tree.branches[value]
    return classify(example, branch)


examples = [
    ['sunny', 'hot', 'high', 'weak', 'no'],
    ['sunny', 'hot', 'high', 'strong', 'no'],
    ['overcast', 'hot', 'high', 'weak', 'yes'],
    ['rainy', 'mild', 'high','weak', 'yes']]

attributes = [0, 1, 2, 3]

default = 'yes'


tree = aq11(examples, attributes, default)


example = ['overcast', 'mild', 'weak', 'strong']
classification = classify(example, tree)
print(classification)  # Output: 'yes'