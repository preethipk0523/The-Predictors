#!/usr/bin/env python3
"""
Generate data/questions.json according to Assignment 1 specifications.
Contains 45 questions across 5 categories with comprehensive metadata.
"""

import json
from pathlib import Path

SOURCE_DOC = "crystal1 (1).pdf"

questions = [
    # ==========================================
    # CATEGORY 1: FACTUAL (12 questions >= 10)
    # ==========================================
    {
        "question": "What are the four levels of the data-to-wisdom hierarchy described in Chapter 1?",
        "expected_answer": "The four levels from bottom to top are Data, Information, Knowledge, and Wisdom.",
        "source": SOURCE_DOC,
        "page": 8,
        "category": "factual",
        "difficulty": "easy",
        "priority": True,  # Priority 1
        "expected_chunks": [{"source": SOURCE_DOC, "page": 8}],
    },
    {
        "question": "What are the three kinds of atomic data described in Chapter 3?",
        "expected_answer": "The three kinds of atomic data are numeric data, categorical data, and text data.",
        "source": SOURCE_DOC,
        "page": 19,
        "category": "factual",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 19}],
    },
    {
        "question": "What is the fundamental difference between interval and ratio scales of measurement according to Chapter 6?",
        "expected_answer": "A ratio scale possesses a true, non-arbitrary zero point representing the complete absence of the measured quantity, allowing meaningful ratios and multiplication, whereas an interval scale has an arbitrary zero point.",
        "source": SOURCE_DOC,
        "page": 54,
        "category": "factual",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 54}],
    },
    {
        "question": "What are the three aggregate data types introduced in Chapter 7?",
        "expected_answer": "The three aggregate data types are arrays, associative arrays, and tables.",
        "source": SOURCE_DOC,
        "page": 59,
        "category": "factual",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 59}],
    },
    {
        "question": "What does vectorized calculation mean in NumPy and Python?",
        "expected_answer": "Vectorized calculation means applying an arithmetic or logical operation to an entire array or series of values element-by-element simultaneously without writing an explicit loop.",
        "source": SOURCE_DOC,
        "page": 83,
        "category": "factual",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 83}],
    },
    {
        "question": "What is Simpson's Paradox as illustrated in Chapter 16?",
        "expected_answer": "Simpson's Paradox occurs when an apparent trend or pattern observed in aggregated data disappears or reverses direction when the data is partitioned into subgroups based on a confounding variable.",
        "source": SOURCE_DOC,
        "page": 177,
        "category": "factual",
        "difficulty": "hard",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 177}],
    },
    {
        "question": "How is the Interquartile Range (IQR) defined and calculated in univariate exploratory data analysis?",
        "expected_answer": "The Interquartile Range (IQR) is the difference between the 75th percentile (third quartile, Q3) and the 25th percentile (first quartile, Q1), representing the spread of the middle 50% of values: IQR = Q3 - Q1.",
        "source": SOURCE_DOC,
        "page": 156,
        "category": "factual",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 156}],
    },
    {
        "question": "What is the difference between .loc and .iloc when indexing a Pandas DataFrame or Series?",
        "expected_answer": ".loc performs label-based indexing using row or column names/labels, whereas .iloc performs integer-position-based indexing using numeric offsets.",
        "source": SOURCE_DOC,
        "page": 183,
        "category": "factual",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 183}],
    },
    {
        "question": "What does the string method .strip() do in Python?",
        "expected_answer": "The .strip() method removes leading and trailing whitespace characters from both ends of a string.",
        "source": SOURCE_DOC,
        "page": 43,
        "category": "factual",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 43}],
    },
    {
        "question": "What is 'the prior' of a data set in the context of classification?",
        "expected_answer": "The prior is the baseline probability distribution or proportion of each target label in the dataset without considering any predictive features, serving as the benchmark for a classifier.",
        "source": SOURCE_DOC,
        "page": 274,
        "category": "factual",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 274}],
    },
    {
        "question": "Why must a classifier always be evaluated on a separate test set rather than its training set?",
        "expected_answer": "Evaluating on training data gives a deceptively high performance because the model can simply memorize the training examples; testing on unseen data is essential to assess true generalization to new observations.",
        "source": SOURCE_DOC,
        "page": 295,
        "category": "factual",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 295}],
    },
    {
        "question": "What is a labeled example in classification?",
        "expected_answer": "A labeled example is an observation or object of study provided with its known true answer or target variable value, used during supervised training.",
        "source": SOURCE_DOC,
        "page": 267,
        "category": "factual",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 267}],
    },

    # ==========================================
    # CATEGORY 2: CROSS-REFERENCE (9 questions >= 8, covering 5 sub-types)
    # ==========================================
    {
        "question": "How does the conceptual idea of an associative array in Chapter 7 transition into Pandas Series in Chapter 11?",
        "expected_answer": "Chapter 7 defines an associative array conceptually as key-value pairs without position indexing, while Chapter 11 implements this in Pandas as a Series object where keys become the Series index.",
        "source": SOURCE_DOC,
        "page": [59, 109],
        "category": "cross-reference",
        "sub_type": "same_document",
        "difficulty": "medium",
        "priority": True,  # Priority 2
        "expected_chunks": [
            {"source": SOURCE_DOC, "page": 59},
            {"source": SOURCE_DOC, "page": 109},
        ],
    },
    {
        "question": "How does the definition of categorical atomic data in Chapter 3 relate to the nominal scale in Chapter 6?",
        "expected_answer": "Chapter 3 introduces categorical data as discrete categories, and Chapter 6 formalizes this as the nominal scale of measurement where categories have no inherent order and only equality comparisons are valid.",
        "source": SOURCE_DOC,
        "page": [19, 50],
        "category": "cross-reference",
        "sub_type": "same_document",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [
            {"source": SOURCE_DOC, "page": 19},
            {"source": SOURCE_DOC, "page": 50},
        ],
    },
    {
        "question": "How do consecutive sections in Chapter 15 contrast the mean and median regarding their sensitivity to outliers?",
        "expected_answer": "Page 154 explains that the mean sums all values and divides by count, making it highly sensitive to extreme outliers, whereas page 155 shows the median is the middle sorted value and remains robust against outliers.",
        "source": SOURCE_DOC,
        "page": [154, 155],
        "category": "cross-reference",
        "sub_type": "adjacent_chunks",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [
            {"source": SOURCE_DOC, "page": 154},
            {"source": SOURCE_DOC, "page": 155},
        ],
    },
    {
        "question": "How does the explanation of box plot components connect between pages 171 and 172?",
        "expected_answer": "Page 171 introduces the central box representing the interquartile range (25th to 75th percentiles) and median line, while page 172 details the whiskers extending 1.5 times the IQR and how points beyond the whiskers are drawn as individual outliers.",
        "source": SOURCE_DOC,
        "page": [171, 172],
        "category": "cross-reference",
        "sub_type": "adjacent_chunks",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [
            {"source": SOURCE_DOC, "page": 171},
            {"source": SOURCE_DOC, "page": 172},
        ],
    },
    {
        "question": "Where is the warning to test classifiers on unseen test data repeated in the book?",
        "expected_answer": "The critical warning never to evaluate a classifier on its training data is first introduced when explaining train/test splitting in Chapter 26 (page 265) and repeated with emphasis in Chapter 29 (page 295) during performance evaluation.",
        "source": SOURCE_DOC,
        "page": [265, 295],
        "category": "cross-reference",
        "sub_type": "repeated",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": [
            {"source": SOURCE_DOC, "page": 265},
            {"source": SOURCE_DOC, "page": 295},
        ],
    },
    {
        "question": "How is the metaphor of variables as named memory storage repeated between Chapters 3 and 4?",
        "expected_answer": "Chapter 3 introduces variables as named containers holding values, and Chapter 4 visualizes this in memory diagrams with variable names pointing directly to memory boxes.",
        "source": SOURCE_DOC,
        "page": [21, 31],
        "category": "cross-reference",
        "sub_type": "repeated",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": [
            {"source": SOURCE_DOC, "page": 21},
            {"source": SOURCE_DOC, "page": 31},
        ],
    },
    {
        "question": "How does the initial concept of decision tree partitioning compare with the recursive splitting algorithm in Chapter 28?",
        "expected_answer": "Chapter 27 gives a high-level overview of classifying data by asking sequential questions, whereas Chapter 28 provides the concrete recursive induction algorithm calculating split counts feature by feature.",
        "source": SOURCE_DOC,
        "page": [275, 287],
        "category": "cross-reference",
        "sub_type": "summary_vs_detail",
        "difficulty": "hard",
        "priority": False,
        "expected_chunks": [
            {"source": SOURCE_DOC, "page": 275},
            {"source": SOURCE_DOC, "page": 287},
        ],
    },
    {
        "question": "How does the early mention of correlation in Chapter 1 differ from the comprehensive treatment in Chapter 20?",
        "expected_answer": "Chapter 1 mentions correlation informally as a cautionary example that correlation does not imply causation, while Chapter 20 formally defines scatter plots, linear trends, and Pearson's correlation coefficient r.",
        "source": SOURCE_DOC,
        "page": [9, 215],
        "category": "cross-reference",
        "sub_type": "summary_vs_detail",
        "difficulty": "hard",
        "priority": False,
        "expected_chunks": [
            {"source": SOURCE_DOC, "page": 9},
            {"source": SOURCE_DOC, "page": 215},
        ],
    },
    {
        "question": "How does the simple accuracy metric described on page 296 conflict with the evaluation of imbalanced datasets on page 297?",
        "expected_answer": "Page 296 states that classification accuracy (fraction of correct predictions) is the standard performance metric, but page 297 explains that in imbalanced datasets (e.g. 99% majority class), raw accuracy is deceptive because a useless classifier guessing only the majority label still scores 99%.",
        "source": SOURCE_DOC,
        "page": [296, 297],
        "category": "cross-reference",
        "sub_type": "contradictory",
        "difficulty": "hard",
        "priority": False,
        "expected_chunks": [
            {"source": SOURCE_DOC, "page": 296},
            {"source": SOURCE_DOC, "page": 297},
        ],
    },

    # ==========================================
    # CATEGORY 3: STRESS TESTS (14 questions >= 10, covering 9 sub-types)
    # ==========================================
    {
        "question": "How does the author explain the difference between a predictor feature and a target variable in classification?",
        "expected_answer": "Predictor features are the inputs or attributes available about an observation used to make an inference, while the target variable is the specific outcome or categorical label being predicted.",
        "source": SOURCE_DOC,
        "page": 267,
        "category": "stress-test",
        "sub_type": "paraphrased",
        "difficulty": "medium",
        "priority": True,  # Priority 3
        "expected_chunks": [{"source": SOURCE_DOC, "page": 267}],
    },
    {
        "question": "What informal nickname does the author use for parentheses () throughout the book?",
        "expected_answer": "The author humorously refers to parentheses () as 'bananas'.",
        "source": SOURCE_DOC,
        "page": 43,
        "category": "stress-test",
        "sub_type": "exact_term",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 43}],
    },
    {
        "question": "Does the Celsius temperature scale represent a ratio scale of measurement?",
        "expected_answer": "No, Celsius is an interval scale, not a ratio scale, because 0 degrees Celsius does not signify the absolute absence of temperature.",
        "source": SOURCE_DOC,
        "page": 54,
        "category": "stress-test",
        "sub_type": "yes_no",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 54}],
    },
    {
        "question": "Can an associative array in Python have duplicate keys according to Chapter 11?",
        "expected_answer": "No, each key in an associative array must be unique so that each key maps unambiguously to its associated value.",
        "source": SOURCE_DOC,
        "page": 109,
        "category": "stress-test",
        "sub_type": "yes_no",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 109}],
    },
    {
        "question": "What technological limitations previously prevented data science from flourishing before the modern era?",
        "expected_answer": "Historically, processing power and storage capacity required to analyze large datasets were not commonly available, and extensive digital datasets had not yet been collected.",
        "source": SOURCE_DOC,
        "page": 7,
        "category": "stress-test",
        "sub_type": "temporal",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 7}],
    },
    {
        "question": "Which mathematical operations are NOT permissible on nominal scale data?",
        "expected_answer": "Arithmetic operations like addition, subtraction, multiplication, division, and mathematical ordering (greater than/less than) are not valid on nominal data; only testing for equality or difference is permitted.",
        "source": SOURCE_DOC,
        "page": 50,
        "category": "stress-test",
        "sub_type": "negation",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 50}],
    },
    {
        "question": "If an evaluation on students_test yields an 87.5% test accuracy, how was this metric derived from the code on page 298?",
        "expected_answer": "By counting the rows where the classifier's predicted label matched the true target label (count = 7) and dividing by the total number of rows in students_test (8), then multiplying by 100.",
        "source": SOURCE_DOC,
        "page": 298,
        "category": "stress-test",
        "sub_type": "multi_hop",
        "difficulty": "hard",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 298}],
    },
    {
        "question": "What are the four scales of measurement in Stevens' hierarchy listed in Chapter 6?",
        "expected_answer": "The four scales in ascending order of mathematical sophistication are Nominal, Ordinal, Interval, and Ratio.",
        "source": SOURCE_DOC,
        "page": 58,
        "category": "stress-test",
        "sub_type": "table_list",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 58}],
    },
    {
        "question": "Which string methods alter the casing of characters as demonstrated on page 43?",
        "expected_answer": "The methods are .upper() (converts to all uppercase), .lower() (converts to all lowercase), and .title() (converts to title case).",
        "source": SOURCE_DOC,
        "page": 43,
        "category": "stress-test",
        "sub_type": "table_list",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 43}],
    },
    {
        "question": "What happens to unmeasured academic duties when faculty evaluation data tracks only publication and teaching ratings?",
        "expected_answer": "Faculty members prioritize and maximize only the tracked metrics, while unmeasured contributions like committee service, student advising, and independent study supervision are minimized in perceived importance.",
        "source": SOURCE_DOC,
        "page": 9,
        "category": "stress-test",
        "sub_type": "edge_of_chunk",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 9}],
    },
    {
        "question": "What's the deal with doing math on strings in Python?",
        "expected_answer": "In Python, strings can be combined with the plus operator (+) for concatenation and multiplied by integers (*) for repetition, but subtraction and division are invalid operations.",
        "source": SOURCE_DOC,
        "page": 38,
        "category": "stress-test",
        "sub_type": "language_variation",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 38}],
    },
    {
        "question": "What is the formula for calculating Gini impurity in Volume One?",
        "expected_answer": "The textbook explains decision tree splitting through purity counts and prediction error, but does not provide the formal mathematical formula for Gini impurity in Volume One.",
        "source": SOURCE_DOC,
        "page": 287,
        "category": "stress-test",
        "sub_type": "on_topic_unanswerable",
        "difficulty": "hard",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 287}],
    },
    {
        "question": "What are the three quantities compared on page 299, their specific values for the student gamer example, and why are they ordered that way?",
        "expected_answer": "The three quantities are classifier performance on the training set (94.1%), performance on the test set (82.5%), and just using the prior on the test set (62.5%); they appear in this descending order because a model performs best on seen data, slightly worse on unseen test data, and both outperform a naive guess using only the baseline prior.",
        "source": SOURCE_DOC,
        "page": 299,
        "category": "stress-test",
        "sub_type": "multi_part",
        "difficulty": "hard",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 299}],
    },
    {
        "question": "What popular superhero quote does the author cite in Chapter 1 regarding data science responsibility?",
        "expected_answer": "The author cites Spiderman: 'with great power comes great responsibility.'",
        "source": SOURCE_DOC,
        "page": 13,
        "category": "stress-test",
        "sub_type": "exact_term",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": [{"source": SOURCE_DOC, "page": 13}],
    },

    # ==========================================
    # CATEGORY 4: OUT-OF-SCOPE (5 questions >= 3)
    # ==========================================
    {
        "question": "What is the backpropagation algorithm for multi-layer neural networks and how are weight gradients computed?",
        "expected_answer": None,
        "source": None,
        "page": None,
        "category": "out-of-scope",
        "difficulty": "easy",
        "priority": True,  # Priority 4
        "expected_chunks": None,
    },
    {
        "question": "What was the official GDP of France in 2023?",
        "expected_answer": None,
        "source": None,
        "page": None,
        "category": "out-of-scope",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": None,
    },
    {
        "question": "How do you bake a classic sourdough bread loaf at home?",
        "expected_answer": None,
        "source": None,
        "page": None,
        "category": "out-of-scope",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": None,
    },
    {
        "question": "Who won the 2024 ICC Men's T20 Cricket World Cup?",
        "expected_answer": None,
        "source": None,
        "page": None,
        "category": "out-of-scope",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": None,
    },
    {
        "question": "What is quantum entanglement and how do qubits achieve superposition?",
        "expected_answer": None,
        "source": None,
        "page": None,
        "category": "out-of-scope",
        "difficulty": "easy",
        "priority": False,
        "expected_chunks": None,
    },

    # ==========================================
    # CATEGORY 5: AMBIGUOUS (5 questions >= 2)
    # ==========================================
    {
        "question": "How do you evaluate performance?",
        "expected_answer": None,
        "source": None,
        "page": None,
        "category": "ambiguous",
        "difficulty": "medium",
        "priority": True,  # Priority 5
        "expected_chunks": None,
    },
    {
        "question": "How do I filter?",
        "expected_answer": None,
        "source": None,
        "page": None,
        "category": "ambiguous",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": None,
    },
    {
        "question": "Which scale is the best?",
        "expected_answer": None,
        "source": None,
        "page": None,
        "category": "ambiguous",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": None,
    },
    {
        "question": "What is the split?",
        "expected_answer": None,
        "source": None,
        "page": None,
        "category": "ambiguous",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": None,
    },
    {
        "question": "Tell me about the numbers.",
        "expected_answer": None,
        "source": None,
        "page": None,
        "category": "ambiguous",
        "difficulty": "medium",
        "priority": False,
        "expected_chunks": None,
    },
]

def main():
    target_path = Path("data/questions.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Validate constraints
    categories = {}
    priorities = 0
    difficulties = {}
    sub_types = {}

    for q in questions:
        cat = q["category"]
        categories[cat] = categories.get(cat, 0) + 1
        if q["priority"]:
            priorities += 1
        diff = q["difficulty"]
        difficulties[diff] = difficulties.get(diff, 0) + 1
        if "sub_type" in q:
            st = q["sub_type"]
            sub_types[st] = sub_types.get(st, 0) + 1

    print(f"Generated {len(questions)} evaluation questions in {target_path}")
    print("\nBreakdown by Category:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
    print(f"\nPriority count: {priorities} (must be exactly 5)")
    print(f"Difficulties: {difficulties}")
    print(f"Sub-types ({len(sub_types)} distinct): {sub_types}")

if __name__ == "__main__":
    main()
