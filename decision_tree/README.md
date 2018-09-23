# Decision Tree Classifier

Decision tree learning uses a decision tree to go from observations about an item to conclusions about the item's target value.

# key aspects

Where to split and till when to split

1. Entropy: - Entropy is degree of randomness of elements or in other words it is measure of impurity.
2. Impurity: - Impurity is when we have a traces of one class division into other.
3. Information Gained: - 
```
Information Gain (n) =
  Entropy(x) — ([weighted average] * entropy(children for feature))
```

# Advantage of Decision Tree

1. Simple to understand and interpret.
2. Able to handle both numerical and categorical data.
3. Requires little data preparation.
4. Uses a white box model. If a given situation is observable in a model the explanation for the condition is easily explained by boolean logic.
5. Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.
6. Non-statistical approach that makes no assumptions of the training data or prediction residuals;
7. Performs well with large datasets.
8. Mirrors human decision making more closely than other approaches.
9. In built feature selection.

# Disadvantages

1. Trees can be very non-robust.
2. The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and even for simple concepts.
3. Decision-tree learners can create over-complex trees that do not generalize well from the training data.(over fitting)
4. For data including categorical variables with different numbers of levels, information gain in decision trees is biased in favor of attributes with more levels
