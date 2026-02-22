# ---------- Packages ----------
# install.packages(c("data.table", "Matrix", "xgboost"))
library(data.table)  # only used for fwrite (you can replace with write.csv)
library(Matrix)
library(xgboost)

set.seed(42)

#quantile(train$target, probs = 0.99, na.rm = TRUE)

# ---------- Load data ----------
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test  <- read.csv("test_for_participants.csv", stringsAsFactors = FALSE)

# ---------- Remove high-target observations ----------
stopifnot("target" %in% names(train))
train <- train[train$target < 226.122, ] 

# Keep IDs (after cleaning train)
train_id <- train$id
test_id  <- test$id

# ---------- Handle timestamp BEFORE defining features ----------

TIME_COL <- "delivery_start"  # use delivery_start as the main timestamp

stopifnot(TIME_COL %in% names(train),
          TIME_COL %in% names(test))

for (df_name in c("train", "test")) {
  df <- get(df_name)
  
  df[[TIME_COL]] <- as.POSIXct(
    df[[TIME_COL]],
    format = "%Y-%m-%d %H:%M:%S",
    tz = "UTC"
  )
  
  df$year      <- as.integer(format(df[[TIME_COL]], "%Y"))  # <--- NEW
  df$hour      <- as.integer(format(df[[TIME_COL]], "%H"))
  df$wday      <- as.integer(format(df[[TIME_COL]], "%u"))
  df$month     <- as.integer(format(df[[TIME_COL]], "%m"))
  df$day       <- as.integer(format(df[[TIME_COL]], "%d"))
  df$day_of_yr <- as.integer(format(df[[TIME_COL]], "%j"))
  
  df$is_weekend <- as.integer(df$wday %in% c(6, 7))
  
  df$hour_sin <- sin(2 * pi * df$hour / 24)
  df$hour_cos <- cos(2 * pi * df$hour / 24)
  df$wday_sin <- sin(2 * pi * df$wday / 7)
  df$wday_cos <- cos(2 * pi * df$wday / 7)
  
  df$is_peak_hour <- as.integer(df$hour >= 7 & df$hour <= 20)
  
  df[[TIME_COL]]      <- NULL
  if ("delivery_end" %in% names(df)) {
    df[["delivery_end"]] <- NULL
  }
  
  assign(df_name, df, envir = .GlobalEnv)
}

# ---------- Create lag-1 target feature on train ----------
# Use chronological order in the train set
idx <- order(train$year, train$day_of_yr, train$hour)

train$lag1 <- NA_real_
train$lag1[idx] <- c(NA, train$target[idx[-length(idx)]])

# Placeholder column in test (will be filled recursively when predicting)
test$lag1 <- NA_real_

# ---------- Basic preprocessing ----------
target_col <- "target"
id_col     <- "id"

feature_cols <- setdiff(names(train), c(target_col, id_col))

# 1) Detect column types

char_cols <- feature_cols[
  sapply(train[feature_cols], function(x) is.character(x) || is.factor(x))
]

num_cols <- feature_cols[
  sapply(train[feature_cols], is.numeric)
]

# 2) Handle categorical: replace NA / "" with "missing" and align levels

for (col in char_cols) {
  train[[col]] <- as.character(train[[col]])
  test[[col]]  <- as.character(test[[col]])
  
  train[[col]][is.na(train[[col]]) | train[[col]] == ""] <- "missing"
  test[[col]][is.na(test[[col]])  | test[[col]]  == ""] <- "missing"
  
  all_levels <- unique(c(train[[col]], test[[col]]))
  
  train[[col]] <- factor(train[[col]], levels = all_levels)
  test[[col]]  <- factor(test[[col]],  levels = all_levels)
}

# 3) Numeric imputation with medians from train

medians <- sapply(train[num_cols], function(x) median(x, na.rm = TRUE))

for (col in num_cols) {
  train[[col]][is.na(train[[col]])] <- medians[[col]]
  test[[col]][is.na(test[[col]])]   <- medians[[col]]
}

# ---------- Sparse design matrices ----------
form <- as.formula(
  paste0("~ ", paste(feature_cols, collapse = " + "), " - 1")
)

X_train <- sparse.model.matrix(form, data = train)
y_train <- train[[target_col]]

X_test <- sparse.model.matrix(form, data = test)

# ---------- Train/validation split (time-based) ----------
n <- nrow(train)

# order rows chronologically: first by year, then by day of year, then by hour
ord <- order(train$year, train$day_of_yr, train$hour)

cutoff <- floor(0.99999 * n)  # first 80% of time for training, last 20% for validation #0.99999

train_idx <- ord[1:cutoff]
valid_idx <- ord[(cutoff + 1):n]

dtrain <- xgb.DMatrix(data = X_train[train_idx, ], label = y_train[train_idx])
dvalid <- xgb.DMatrix(data = X_train[valid_idx, ], label = y_train[valid_idx])

watchlist <- list(train = dtrain, valid = dvalid)

# ---------- XGBoost model ----------
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.05,
  max_depth = 8,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 5000,
  watchlist = watchlist,
  early_stopping_rounds = 100,
  verbose = 1
)

# ---------- Evaluate on validation set ----------
valid_pred <- predict(model, dvalid)

valid_results <- data.table(
  id        = train_id[valid_idx],
  actual    = y_train[valid_idx],
  predicted = valid_pred,
  error     = valid_pred - y_train[valid_idx],
  abs_error = abs(valid_pred - y_train[valid_idx])
)

setorder(valid_results, -abs_error)
#fwrite(valid_results, "valid_predictions_vs_actual.csv")
print(head(valid_results))

# ---------- Predict on test with recursive lag-1 ----------
n_test   <- nrow(test)
ord_test <- order(test$year, test$day_of_yr, test$hour)

pred <- numeric(n_test)

# Find the last observed (most recent) target in the train set
ord_train <- order(train$year, train$day_of_yr, train$hour)
last_train_target <- y_train[ord_train[length(ord_train)]]

# Work on a copy so we can safely modify lag1
test_rec <- test

for (j in seq_along(ord_test)) {
  i <- ord_test[j]
  
  if (j == 1) {
    # First test row: lag1 is last observed target from the train set
    test_rec$lag1[i] <- last_train_target
  } else {
    # Subsequent test rows: lag1 is the previous prediction in time
    prev_i <- ord_test[j - 1]
    test_rec$lag1[i] <- pred[prev_i]
  }
  
  # Build design matrix for this single row with the updated lag1
  Xi  <- sparse.model.matrix(form, data = test_rec[i, , drop = FALSE])
  dXi <- xgb.DMatrix(data = Xi)
  
  pred[i] <- predict(model, dXi)
}

submission <- data.table(
  id     = test_id,
  target = pred
)

fwrite(submission, "submission.csv")
print(head(submission))


