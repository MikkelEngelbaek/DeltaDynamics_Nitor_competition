# ============================================================
# Detect IDs with very big target without seeing targets at prediction time
# "Very big" := target > 1000 OR in top 0.3% (>= 99.7% quantile)
# ============================================================

library(data.table)
library(Matrix)
library(xgboost)

set.seed(42)

# ---------- Load data ----------
train_raw <- fread("train.csv")              # or: fread("/mnt/data/train (1).csv")
test_raw  <- fread("test_for_participants.csv")

target_col <- "target"
id_col     <- "id"
stopifnot(target_col %in% names(train_raw))

# ---------- Drop rows with missing target in TRAIN ----------
train_clean <- train_raw[!is.na(get(target_col))]
cat("Dropped", nrow(train_raw) - nrow(train_clean),
    "rows with NA target from train.\n")

train_id <- train_clean[[id_col]]
test_id  <- test_raw[[id_col]]

# ---------- Define "extreme" targets in TRAIN ----------
extreme_threshold_value <- 1000      # 1) absolute threshold
extreme_quantile        <- 0.99     # 2) top 0.3% quantile

q_extreme <- quantile(
  train_clean[[target_col]],
  probs = extreme_quantile,
  na.rm = TRUE
)

extreme_label <- as.integer(
  train_clean[[target_col]] > extreme_threshold_value |
    train_clean[[target_col]] >= q_extreme
)

cat("Share of 'extreme' observations in train_clean:",
    mean(extreme_label), "\n")

# ---------- Build feature sets (no id, no target) ----------
feature_cols <- setdiff(names(train_clean), c(id_col, target_col))

train_feat <- train_clean[, ..feature_cols]
test_feat  <- test_raw[,    ..feature_cols]

# Detect categorical vs numeric
char_cols <- feature_cols[
  sapply(train_feat, function(x) is.character(x) || is.factor(x))
]

num_cols <- feature_cols[
  sapply(train_feat, is.numeric)
]

cat("Numeric features:", length(num_cols),
    " | Categorical features:", length(char_cols), "\n")

# ---------- Handle numeric NAs (impute) ----------
# Use train medians; apply same to test
drop_num <- c()

for (col in num_cols) {
  v_train <- train_feat[[col]]
  
  # If column is entirely NA in train, mark for dropping
  if (all(is.na(v_train))) {
    drop_num <- c(drop_num, col)
    next
  }
  
  med <- median(v_train, na.rm = TRUE)
  if (!is.finite(med)) med <- 0
  
  # Replace NAs in train and test with train median
  train_feat[[col]][is.na(train_feat[[col]])] <- med
  if (col %in% names(test_feat)) {
    test_feat[[col]][is.na(test_feat[[col]])]  <- med
  }
}

if (length(drop_num) > 0) {
  cat("Dropping numeric columns with all NA in train:", paste(drop_num, collapse = ", "), "\n")
  train_feat[, (drop_num) := NULL]
  test_feat[,  (drop_num) := NULL]
  num_cols <- setdiff(num_cols, drop_num)
}

# ---------- Harmonise categorical levels between TRAIN and TEST ----------
for (col in char_cols) {
  # Convert to character first
  train_feat[[col]] <- as.character(train_feat[[col]])
  test_feat[[col]]  <- as.character(test_feat[[col]])
  
  # Replace NA / "" with "missing"
  train_feat[[col]][is.na(train_feat[[col]]) | train_feat[[col]] == ""] <- "missing"
  test_feat[[col]][is.na(test_feat[[col]])  | test_feat[[col]]  == ""] <- "missing"
  
  # Joint levels
  all_levels <- sort(unique(c(train_feat[[col]], test_feat[[col]])))
  
  train_feat[[col]] <- factor(train_feat[[col]], levels = all_levels)
  test_feat[[col]]  <- factor(test_feat[[col]],  levels = all_levels)
}

# ---------- Design matrices (one-hot, no NA left) ----------
train_df <- as.data.frame(train_feat)
test_df  <- as.data.frame(test_feat)

X_train <- model.matrix(~ . - 1, data = train_df)
X_test  <- model.matrix(~ . - 1, data = test_df)

cat("Final feature matrix shape:",
    "train:", dim(X_train)[1], "x", dim(X_train)[2],
    "| test:", dim(X_test)[1], "x", dim(X_test)[2], "\n")

# Sanity checks
if (!identical(colnames(X_train), colnames(X_test))) {
  stop("Column names of X_train and X_test do not match. Check factor handling.")
}
if (nrow(X_train) != length(extreme_label)) {
  stop(sprintf("Label length (%d) != nrow(X_train) (%d)",
               length(extreme_label), nrow(X_train)))
}

# ---------- XGBoost model for 'extreme' classification ----------
dtrain <- xgb.DMatrix(
  data  = X_train,
  label = extreme_label
)

params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  max_depth        = 6,
  eta              = 0.1,
  subsample        = 0.8,
  colsample_bytree = 0.8
)

watchlist <- list(train = dtrain)

bst <- xgb.train(
  params                = params,
  data                  = dtrain,
  nrounds               = 300,
  watchlist             = watchlist,
  early_stopping_rounds = 20,
  verbose               = 1
)

cat("Best iteration:", bst$best_iteration, "\n")

# ---------- Predicted probabilities ----------
pred_train_prob <- predict(bst, X_train)
pred_test_prob  <- predict(bst, X_test)

# ============================================================
# Summary of TEST predicted probabilities
# ============================================================

test_prob_dt <- data.table(
  id           = test_id,
  prob_extreme = pred_test_prob
)

# Print basic statistics
cat("\nSummary of predicted extreme probabilities on TEST:\n")
print(summary(test_prob_dt$prob_extreme))

# Quantile overview
cat("\nQuantiles of TEST extreme probabilities:\n")
print(quantile(test_prob_dt$prob_extreme,
               probs = c(0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1),
               na.rm = TRUE))

# Count of low-probability cases
test_prob_dt[, low_1pct  := prob_extreme <= 0.01]
test_prob_dt[, low_5pct  := prob_extreme <= 0.05]
test_prob_dt[, low_10pct := prob_extreme <= 0.10]

cat("\nLow-probability counts in TEST:\n")
cat("  <= 1% probability :", sum(test_prob_dt$low_1pct),  "\n")
cat("  <= 5% probability :", sum(test_prob_dt$low_5pct),  "\n")
cat("  <= 10% probability:", sum(test_prob_dt$low_10pct), "\n")

# Save full probability file
fwrite(test_prob_dt, "test_extreme_probabilities.csv")
cat("\nSaved: test_extreme_probabilities.csv\n")

# ---------- Choose probability cutoff for 'extreme' ----------
extreme_prob_cutoff <- 0.35
cat("Using probability cutoff:", extreme_prob_cutoff, "\n")

train_flag <- pred_train_prob >= extreme_prob_cutoff
test_flag  <- pred_test_prob  >= extreme_prob_cutoff

extreme_train_ids <- data.table(
  id           = train_id[train_flag],
  prob_extreme = pred_train_prob[train_flag]
)[order(-prob_extreme)]

extreme_test_ids <- data.table(
  id           = test_id[test_flag],
  prob_extreme = pred_test_prob[test_flag]
)[order(-prob_extreme)]

cat("Flagged", nrow(extreme_train_ids), "train IDs and",
    nrow(extreme_test_ids), "test IDs as potentially very big targets.\n")

# ---------- Compare guesses to actuals on TRAIN ----------
cm <- table(
  Predicted = as.integer(train_flag),
  Actual    = extreme_label
)
print(cm)

TP <- if ("1" %in% rownames(cm) && "1" %in% colnames(cm)) cm["1", "1"] else 0
TN <- if ("0" %in% rownames(cm) && "0" %in% colnames(cm)) cm["0", "0"] else 0
FP <- if ("1" %in% rownames(cm) && "0" %in% colnames(cm)) cm["1", "0"] else 0
FN <- if ("0" %in% rownames(cm) && "1" %in% colnames(cm)) cm["0", "1"] else 0

accuracy  <- (TP + TN) / sum(cm)
precision <- ifelse((TP + FP) > 0, TP / (TP + FP), NA_real_)
recall    <- ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_)
F1        <- ifelse(is.na(precision) || is.na(recall) ||
                      (precision + recall) == 0,
                    NA_real_,
                    2 * precision * recall / (precision + recall))

cat("\nPerformance on TRAIN (extreme vs not):\n")
cat("  Accuracy :", round(accuracy, 4), "\n")
cat("  Precision:", round(precision, 4), "\n")
cat("  Recall   :", round(recall, 4), "\n")
cat("  F1 score :", round(F1, 4), "\n")

cat("\nMean predicted probability by actual class:\n")
print(tapply(pred_train_prob, extreme_label, mean))

# ---------- Save results ----------
fwrite(extreme_train_ids, "extreme_train_ids.csv")
fwrite(extreme_test_ids,  "extreme_test_ids.csv")

# ---------- Row-level results for TRAIN ----------
train_results <- data.table(
  id              = train_id,
  target          = train_clean[[target_col]],
  extreme_actual  = extreme_label,                 # 1 = actually extreme, 0 = not
  prob_extreme    = pred_train_prob,               # model probability
  extreme_pred    = as.integer(train_flag)         # 1 = predicted extreme, 0 = not
)

conf_summary <- data.table(
  Metric = c("True Positives", "False Positives", "False Negatives", "True Negatives"),
  Count  = c(TP,              FP,               FN,               TN)
)[
  , Percentage := round(100 * Count / sum(Count), 2)
]

print(conf_summary)

# Save full row-level comparison for train
fwrite(train_results, "train_extreme_predictions.csv")


