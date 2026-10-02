# Applied AI
- The Applied AI-Machine Learning course provides a comprehensive curriculum covering foundational programming, data science, machine learning, and deep learning through a series of modular chapters.
- The course structure is organized as follows:
	  * **Fundamentals**: Module 1 introduces Python for data science and SQL for database management.
	  * **EDA and Statistics**: Module 2 covers exploratory data analysis (EDA), linear algebra, and probability and statistics. Dimensionality reduction techniques, including Principal Component Analysis (PCA) and t-SNE, are also addressed.
	  * **Foundations of NLP and ML**: Module 3 focuses on text processing, classification, and regression, covering algorithms such as k-Nearest Neighbors, Naive Bayes, and Logistic Regression.
	  * **Supervised Learning**: Module 4 explores advanced models including Support Vector Machines (SVM), Decision Trees, and Ensemble methods like Random Forest and Gradient Boosting.
	  * **Feature Engineering & Deployment**: Module 5 covers feature engineering and the productionization of ML models, including deployment using AWS.
	  * **Unsupervised Learning**: Module 7 details clustering techniques (e.g., K-Means, DBSCAN) and Recommender Systems, including Matrix Factorization.
	  * **Deep Learning**: Module 8 introduces neural networks, including Multilayer Perceptrons, Convolutional Neural Networks (CNN), RNNs/LSTMs, and Attention models, using TensorFlow and Keras.
	The course emphasizes practical application through extensive real-world case studies for both machine learning and deep learning, covering topics such as Quora question pair similarity, cancer diagnosis, taxi demand prediction, malware detection, self-driving car technology, and music generation.

## Module 1: Fundamentals of Programming
###  1: How to utilise Applied AI Course?
- To successfully complete the course and maximize learning, students should approach the content with a commitment to hard work, patience, and discipline. The following strategies are recommended for managing the 140+ hours of content:
- **Learning Strategies**
	- **Breadth-First Approach:** Adopt a "breadth-first" strategy, which is compared to peeling an onion. Begin by learning high-level concepts and core ideas before digging into deeper, more complex mathematical details.
	- **Sequential Progression:** The course is organized like a story; therefore, it is best to view videos sequentially rather than jumping to random sections.
	- **Note Taking:** Taking personalized notes is mandatory for successful revision. An effective method is to take partial screenshots of equations or core concepts and add your own notes on top.
	- **Feynman Technique:** To master a concept, try explaining it to a virtual friend or writing out the explanation in a notebook. This process helps identify gaps in understanding, which can then be addressed by returning to the original source material.
- **Revision and Discipline**
	- **Spaced Interval Learning:** To retain information, follow a spaced revision schedule:
	    1. **Day 1:** Revise using the Feynman technique.
	    2. **1 Week Later:** Use the provided interview questions at the end of each chapter.
	    3. **1 Month Later:** Apply the Feynman technique once more.
	- **Maintain Consistency:** Establish fixed study hours, such as early morning "golden hours" or designated commute time, to ensure discipline.
	- **Leverage External Resources:** Cultivate "learning to learn" skills by using Google Image Search, Stack Overflow, Quora, and technical blogs to find visual aids and deeper explanations.
- **Support and Communication**

	- **Public Queries:** When you have questions, post them in the video comment section rather than sending private emails. This allows others to benefit from the answer, and your query may have already been addressed there.
	- **Monitor Emails:** Regularly check emails from the course, as they contain progress reports, responses to your queries, and additional reading materials.

- **Assignments and Debugging**

	- **Mandatory Assignments:** These are essential for the job guarantee/cashback program and must be submitted via Google Classroom in both IPython notebook and PDF formats.
	- **Plagiarism:** Plagiarism is strictly prohibited. You may use code snippets as references, but they must be understood and modified to reflect your own work.
	- **Independent Debugging:** When encountering coding errors, copy and search the exact error message on Google or Stack Overflow to find resolutions before requesting assistance.
###  2: Python for Data Science: Introduction
#### Why learn Python?
- The decision to prioritize Python for this course over languages such as C, C++, or Java is based on several key factors.
	- **Ease of Learning**
	    - Python is described as a simple language to learn.
	    - For students or professionals already familiar with C, C++, or Java, picking up Python is considered a "cakewalk".
	- **Extensive Library Ecosystem**
	    - A primary reason for choosing Python is its collection of powerful packages designed for machine learning, artificial intelligence, and data science.
	    - These packages include:
	        - Matplotlib for plotting.
	        - NumPy and SciPy for scientific computing.
	        - Scikit-learn for machine learning tasks.
	        - TensorFlow for deep learning.
	    - The availability of these libraries makes development easier, as it eliminates the need to implement complex, intricate algorithms from scratch.
	- **Interactive Frameworks**
	    - Python utilizes IPython notebooks, which serve as a useful tool for data analysis and modeling.
	    - This interactive framework allows for the integration of code and content.
	    - It runs through web browsers and is easily shareable, making it convenient to run code on various devices.
	- **Versatility Compared to Other Languages**
	    - While languages like R are used in the industry, they are primarily limited to statistical programming.
	    - Python is preferred because it is a general-purpose language.
	    - This general-purpose nature ensures that learning Python provides broader utility for various programming applications beyond just AI and data science.
#### Keywords and Identifiers
- **Keywords in Python**
	Keywords are reserved words that have specific meanings in the Python language and cannot be used for any other purpose.
	- **Definition**: They are specialized words used by the language itself to define its structure and logic.
	- **Usage Restrictions**: Keywords cannot be used as variable names, function names, or any other type of identifier.
	- **Case Sensitivity**: Python keywords are case-sensitive.
	- **Quantity**: There are 33 keywords in Python 3.
	- **Examples**: Common examples include `if`, `else`, `for`, `class`, `raise`, `yield`, and `return`.
	- **How to View Keywords**: You can see the full list of keywords by importing the `keyword` module and printing `keyword.kwlist`.
- **Identifiers in Python**
	Identifiers are the names given to various entities in a program, such as variables, functions, or classes.
	- **Naming Rules**:
	    - **Permitted Characters**: Identifiers can be a combination of lowercase letters (a-z), uppercase letters (A-Z), digits (0-9), and underscores (`_`).
	    - **Starting Character**: They **cannot** begin with a digit. For example, `12_abc` is invalid, but `abc_12` is valid.
	    - **Reserved Words**: Keywords cannot be used as identifiers. For instance, attempting to assign a value to `global` (a keyword) will result in a syntax error.
	    - **Special Symbols**: Special characters like `@`, `#`, `$`, `%`, or `&` are not allowed within an identifier.
	- **Error Handling**: Python provides readable and easy-to-interpret error messages when these rules are violated, which assists in debugging.
#### Comments, Indentation, and Statements
- Python programming emphasizes readability and specific structural formatting. Below is a summary of the concepts covered:
	1.  **Comments**
	      * Comments are non-executed statements used to provide explanations, making code more readable for humans.
	      * Single-line comments are created by starting a line with a hash symbol (\#).
	      * Multi-line comments can be implemented by using a hash on each line or by enclosing text in triple quotes (single or double), which are typically used for multi-line strings.
	2.  **Indentation and Structure**
	      * Unlike languages like C, C++, or Java that rely on curly braces to define blocks of code, Python utilizes indentation to manage code flow.
	      * Everything indented at the same level is considered part of the preceding block, such as a loop.
	      * It is recommended to use four spaces for indentation. Using tabs is discouraged because they may be interpreted differently across various text editors.
	      * Improper indentation will trigger an "indentation error".
	3.  **Code Readability**
	      * Indentation is critical for maintaining code that is easy to read and follow.
	      * Although it is possible to write multiple statements on a single line using a colon (such as within an `if` conditional), doing so is often less readable.
	      * Because others may read the code in the future, maintaining high readability is considered as important as ensuring code correctness.
	4.  **Statements**
	      * Statements are commands executed by the Python interpreter, such as assigning a value to a variable.
      * **Multi-line Statements:** 
	      * If a statement spans multiple lines, a backslash () must be placed at the end of the line to indicate that the statement continues. Alternatively, wrapping the entire statement in parentheses allows it to span multiple lines without needing a backslash.
      * **Multiple Statements on One Line:** 
	      * It is possible to include multiple statements on a single line for brevity, but they must be separated by semicolons (;). Omitting the semicolon in this context will cause an error.

#### Variables and Datatypes in Python
- **Variables in Python**
	- **Definition**: A variable is a reserved memory location used to store values.
	- **Naming Rules**: Variable names follow the same rules as identifiers: they cannot start with a digit or contain special characters like `#` or `%`.
	- **Dynamic Typing**: Unlike languages like C, Python does not require explicit data type declarations. It automatically identifies the data type based on the value assigned.
	- **Memory Efficiency**: Multiple variables pointing to the same value (e.g., `x = 3` and `y = 3`) will initially share the same memory location to save space.
	- **Assignment Methods**:
	    - **Simple**: `a = 10`.
	    - **Multiple**: `a, b, c = 10, 5.5, "ML"` assigns values to multiple variables in one line.
	    - **Same Value**: `a = b = c = "AI"` assigns one value to several variables at once.
- **Standard Data Types**
	Python treats everything as an **object**. You can use the `type()` function to check any variable's data type.

| Data Type            | Description                              | Key Characteristics                                                                  |
| -------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------ |
| **Integer (`int`)**  | Whole numbers.                           | Identified automatically.                                                            |
| **Float (`float`)**  | Decimal numbers.                         | Identified by the presence of a decimal.                                             |
| **Complex**          | Mathematical numbers in `a + bj` format. | Can be verified using `isinstance(variable, complex)`.                               |
| **Boolean (`bool`)** | Logical values: `True` or `False`.       | These are keywords and must be capitalized.                                          |
| **String (`str`)**   | Sequence of Unicode characters.          | Can use single (`'`), double (`"`), or triple quotes (`"""`) for multi-line strings. |

- **Advanced Data Structures**
	- **Lists `[]`**: Ordered, **mutable** (changeable) collections that can store mixed data types. They use zero-based indexing.
	- **Tuples `()`**: Ordered, **immutable** (unchangeable) collections. Once assigned, their elements cannot be modified.
	- **Sets `{}`**: Unordered collections of **unique** items. They do not support indexing and automatically remove duplicate values.
	- **Dictionaries `{}`**: Key-value stores (similar to hash tables). Values are accessed via unique keys rather than numeric indices.
- **Important Functions & Operations**
	- **Indexing & Slicing**: Python uses zero-based indexing. Negative indexing (e.g., `-1`) allows accessing elements from the end of a sequence. Slicing (e.g., `[5:10]`) extracts a specific range.
	- **Type Conversion**: Functions like `int()`, `float()`, and `str()` convert between types. This is often required for tasks like concatenating a string with a number.
	- **Complex Conversion**: You can convert between structures, such as turning a string into a list of characters using `list()`.
###  3: Python for Data Science: Data Structures
###  4: Python for Data Science: Functions
###  5: Python for Data Science: Functions
###  6: Python for Data Science: Matplotlib
###  7: Python for Data Science: Pandas
###  8: Computational Complexity: an Introduction
###  9: SQL
## Module 2: Data Science: Exploratory Data Analysis and Data Visualization
###  1: Plotting for exploratory data analysis (EDA)
###  2: Linear Algebra
###  3: Probability and Statistics
###  4: Interview Questions on Probability and Statistics
###  5: Dimensionality reduction and Visualization:
###  6: Principal Component Analysis.
###  7: T-distributed stochastic neighbourhood embedding (t-SNE)
###  8: Interview Questions on Dimensionality Reduction
## Module 3: Foundations of Natural Language Processing and Machine Learning
###  1: Real world problem: Predict rating given product reviews on Amazon.
###  2: Classification and Regression Models: K-Nearest Neighbours
###  3: Interview Questions on k-NN
###  4: Classification algorithms in various situations:
###  5: Performance measurement of models:
###  6: Interview Questions on Performance Measurement models.
###  7: Naive Bayes
###  8: Logistic Regression:
###  9: Linear Regression.
###  10: Solving optimization problems
###  11: Interview questions on Logistic Regression and Linear Regression  
## Module 4: Machine Learning- II (Supervised Learning Models)
###  1: Support Vector Machines (SVM)
###  2: Interview Questions on Support Vector Machine
###  3: Decision Trees
###  4: Interview Questions on Decision Trees.
###  5: Ensemble Models:
## Module 5: Feature Engineering, Productionisation and deployment of ML Models
###  1: Featurizations and Feature engineering.
###  2: Miscellaneous Topics
## Module 6: Machine Learning Real-World Case Studies
###  1: Case study 1: Quora Question pair similarity problem
###  2: Case study 2: Personalized Cancer Diagnosis
###  3: Case Study 3: Facebook Friend Recommendation using Graph mining.
###  4: Case study 4:Taxi demand prediction in New York City.
###  5: Case Study 5: Stack overflow Tag Predictor
###  6: Case Study 6: Microsoft Malware Detection
###  7: Case study 7: AD-Click Prediction
## Module 7: Data Mining (Unsupervised Learning) and Recommender Systems + Real -world Case Studies
###  1: Unsupervised learning/Clustering
###  2: Hierarchical clustering Technique
###  3: DBSCAN (Density based clustering)
###  4: Recommender Systems and Matrix Factorization.
###  5: Interview Questions on Recommender Systems and Matrix Factorization.
###  6: Case Study 8: Amazon Fashion Discovery Engine
###  7: Case Study 9: Netflix Movie Recommendation System
## Module 8: Neutral Networks, Computer Vision and Deep Learning
###  1: Deep Learning: Neural Networks.
###  2: Deep Learning: Deep Multi-layer perceptions
###  3: Deep Learning: TensorFlow and Keras.
###  4: Deep Learning: Convolutional Neural Nets.
###  5: Deep Learning: Long Short-Term Memory (LSTMS)
###  6: Deep Learning generative Adversarial Networks (GANs).
###  7: Encoder-Decoder Models
###  8: Attention Models in Deep Learning
###  9: Image Segmentation
###  10: Interview Questions on Deep Learning
## Module 9: Deep Learning Real-World Case Studies
###  1: Case Study 11: Human Activity Recognition.
###  2: Case Study 10: Self-Driving Car
###  3: Case Study 12: Music Generation using Deep Learning.
###  4: Interview Questions