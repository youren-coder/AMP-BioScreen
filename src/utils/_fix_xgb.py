"""Fix early_stopping_rounds for xgboost 3.x"""
import os
import sys, os
_utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_utils_dir, '..'))
from paths import PROJECT_ROOT, DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR

p = PROJECT_ROOT / "src/06_train_xgboost.py"
with open(p, encoding="utf-8") as f:
    src = f.read()

# Fix 1: Move early_stopping_rounds from fit() to constructor
src = src.replace(
    'eval_metric="auc",\n                random_state=42,',
    'eval_metric="auc",\n                early_stopping_rounds=30,\n                random_state=42,'
)
# Fix 2: Remove from fit()
src = src.replace(
    'early_stopping_rounds=30,\n                verbose=False,',
    'verbose=False,'
)
# Fix 3: Save final model with best_ntree_limit
old = "model.save_model(str(FEATURE_DIR / "
new = "model.save_model(str(FEATURE_DIR / "
# Actually, best_ntree_limit is automatically handled by XGBClassifier

with open(p, "w", encoding="utf-8") as f:
    f.write(src)
compile(src, p, "exec")
print("Fixed!")
print(f"{len(src)} chars, {src.count(chr(10))} lines")
