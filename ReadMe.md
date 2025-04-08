# 📦 spamtorch: An Spam Classifier Using torch in R

Welcome to `spamtorch` — a **production-ready, fully-documented, and unit-tested spam classifier** built using `torch` in R. This package is designed not only for performance, but also for clarity, modularity, reproducibility, and extensibility —ideal for research and real-world applications alike.

---

## Features of the ML model-

### Clean Data Loading with Sanity Checks
- Automatically downloads and parses the classic [UCI Spam Dataset](https://hastie.su.domains/ElemStatLearn/datasets/spam.data).
- Includes robust validation for column count and dataset size.
- Splits data into training/testing with reproducibility and normalization built-in.

### 🧠 Powerful Neural Network Architecture
- Deep `torch`-based feedforward network with:
  - Batch Normalization for faster, stable convergence.
  - Dropout layers to prevent overfitting.
  - ReLU activations.
- Automatically handles **imbalanced datasets** using **class weighting** in loss function.

### 📉 Early Stopping & Learning Rate Scheduling
- Implements **early stopping** with configurable patience to avoid overfitting.
- Uses **One-Cycle LR Scheduler** for fast convergence.

### 📈 Metrics-Rich Evaluation
- Outputs:
  - Confusion Matrix
  - Accuracy
  - Sensitivity, Specificity, Balanced Accuracy
  - Kappa Statistic
  - ROC Curve
- Perfect for research use, thesis benchmarking, or GSoC reports.

### Designed as a Proper R Package
- Fully documented with **Roxygen2**
- Comes with **unit tests using `testthat`**
- Includes a **vignette** for reproducible usage
- Lightweight dependencies, fast install

### Beginner-Friendly Yet Scalable
- Clear, commented, and modular R code
- Makes deep learning in R accessible to beginners while following best practices
- Easily extensible to add more layers, models, or datasets

---

## 🔧 Installation

```r
# Install devtools if needed
install.packages("devtools")

# Install from local directory (assuming you cloned or created it on Desktop)
devtools::install("path/to/spamtorch")

Usage Example

library(spamtorch)

# Load and preprocess data
data <- load_spam_data()
datasets <- prepare_datasets(data)

# Create data loaders
train_dl <- dataloader(datasets$train, batch_size = 64, shuffle = TRUE)
test_dl <- dataloader(datasets$test, batch_size = 64)

# Train and evaluate the model
model <- create_model()
model <- train_model(model, train_dl, test_dl)
results <- evaluate_model(model, test_dl)

# Print the confusion matrix
print(results$confusion_matrix)


Example Output
Confusion Matrix and Statistics

          Reference
Prediction   1   2
         1 511  36
         2  36 338

               Accuracy : 0.9218
               Kappa    : 0.8379
         Balanced Acc.  : 0.9190
         Sensitivity    : 0.9037
         Specificity    : 0.9342
