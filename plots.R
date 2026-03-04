# ==============================
# 0. Setup
# ==============================
library(ggplot2)
library(dplyr)
library(tidyr)

# Create output directory for plots
out_dir <- "plots"
if (!dir.exists(out_dir)) dir.create(out_dir)

# ==============================
# 1. Load data
# ==============================
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test  <- read.csv("test_for_participants.csv", stringsAsFactors = FALSE)

stopifnot("target" %in% names(train))
stopifnot("id" %in% names(train))

target_col <- "target"
id_col     <- "id"

# ==============================
# 2. Identify feature types
# ==============================
feature_cols <- setdiff(names(train), c(target_col, id_col))

# Numeric features
num_cols <- feature_cols[sapply(train[feature_cols], is.numeric)]

# Categorical (character / factor) features
cat_cols <- feature_cols[
  sapply(train[feature_cols], function(x) is.character(x) || is.factor(x))
]

# For safety, convert character categoricals to factors
for (col in cat_cols) {
  train[[col]] <- as.factor(train[[col]])
}

# ==============================
# 3. Correlation heatmap (numeric, including target)
# ==============================
num_for_cor <- c(num_cols, target_col)
num_for_cor <- num_for_cor[num_for_cor %in% names(train)]

if (length(num_for_cor) >= 2) {
  cor_data <- train[, num_for_cor]
  cor_mat  <- cor(cor_data, use = "pairwise.complete.obs")
  
  cor_df <- data.frame(
    Var1  = rep(rownames(cor_mat), times = ncol(cor_mat)),
    Var2  = rep(colnames(cor_mat), each  = nrow(cor_mat)),
    value = as.vector(cor_mat),
    row.names = NULL
  )
  
  p_cor <- ggplot(cor_df, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    geom_text(aes(label = round(value, 2), color = abs(value) > 0.7), size = 3) +
    scale_color_manual(values = c("black", "white"), guide = "none") +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                         midpoint = 0, limit = c(-1, 1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid = element_blank()) +
    labs(title = "Correlation heatmap (numeric features + target)",
         x = NULL, y = NULL, fill = "cor")
  
  ggsave(filename = file.path(out_dir, "correlation_heatmap.png"),
         plot = p_cor, width = 8, height = 6, dpi = 300)
}

# ==============================
# 4. Market correlation heatmap
# ==============================
if (all(c("market", "delivery_start") %in% names(train))) {
  train$delivery_start <- as.POSIXct(train$delivery_start)
  train$market         <- as.factor(train$market)
  
  train_agg <- train %>%
    group_by(delivery_start, market) %>%
    summarise(target = mean(target, na.rm = TRUE), .groups = "drop")
  
  wide <- train_agg %>%
    tidyr::pivot_wider(
      id_cols     = delivery_start,
      names_from  = market,
      values_from = target
    )
  
  market_cols <- setdiff(names(wide), "delivery_start")
  wide_num    <- as.data.frame(wide[, market_cols])
  wide_num[]  <- lapply(wide_num, as.numeric)
  cor_markets <- cor(wide_num, use = "pairwise.complete.obs")
  
  cor_df_markets <- data.frame(
    Market1   = rep(rownames(cor_markets), times = ncol(cor_markets)),
    Market2   = rep(colnames(cor_markets), each  = nrow(cor_markets)),
    value     = as.vector(cor_markets),
    row.names = NULL
  )
  
  p_market_cor <- ggplot(cor_df_markets, aes(x = Market1, y = Market2, fill = value)) +
    geom_tile() +
    geom_text(aes(label = round(value, 2), color = abs(value) > 0.7), size = 3) +
    scale_color_manual(values = c("black", "white"), guide = "none") +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                         midpoint = 0, limit = c(-1, 1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid = element_blank()) +
    labs(title = "Time-aligned correlation of target between markets",
         x = "Market", y = "Market", fill = "cor")
  
  ggsave(filename = file.path(out_dir, "correlation_between_markets.png"),
         plot = p_market_cor, width = 7, height = 6, dpi = 300)
}

# ==============================
# 5. Histograms for numeric features
# ==============================
for (col in num_cols) {
  p_hist <- ggplot(train, aes_string(x = col)) +
    geom_histogram(bins = 30, color = "black", alpha = 0.7) +
    theme_minimal() +
    labs(title = paste("Histogram of", col), x = col, y = "Count")
  
  ggsave(filename = file.path(out_dir, paste0("hist_", col, ".png")),
         plot = p_hist, width = 6, height = 4, dpi = 300)
}

# ==============================
# 6. Scatterplots: target vs numeric features
# ==============================
for (col in num_cols) {
  p_scatter <- ggplot(train, aes_string(x = col, y = target_col)) +
    geom_point(alpha = 0.4) +
    theme_minimal() +
    labs(title = paste("Target vs", col), x = col, y = target_col)
  
  ggsave(filename = file.path(out_dir, paste0("scatter_target_vs_", col, ".png")),
         plot = p_scatter, width = 6, height = 4, dpi = 300)
}

# ==============================
# 7. Boxplots: target vs categorical features
# ==============================
max_levels <- 20

for (col in cat_cols) {
  n_levels <- length(levels(train[[col]]))
  if (n_levels <= max_levels && n_levels >= 2) {
    p_box <- ggplot(train, aes_string(x = col, y = target_col)) +
      geom_boxplot(outlier.alpha = 0.3) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = paste("Target by", col), x = col, y = target_col)
    
    ggsave(filename = file.path(out_dir, paste0("box_target_by_", col, ".png")),
           plot = p_box, width = 7, height = 4, dpi = 300)
  }
}

# ==============================
# 8. Done
# ==============================
cat("Plots saved in directory:", out_dir, "\n")