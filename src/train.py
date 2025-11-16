from xgb_model import XGB_model
from utils import get_data

TRAIN_FOLDER = "/home/gleb_siz/ml_training/recipt_parser/data/SROIE2019/train"
TEST_FOLDER = "/home/gleb_siz/ml_training/recipt_parser/data/SROIE2019/test"

CONFIG = {
    'features': [
        "file_aspect_ratio",
        "x_max",
        "token_width",
        "token_heigh",
        "aspect_ratio",
        "row",
        "col",
        "row_rank",
        "col_rank",
        "has_total_keyword_in_row",
        "tokens_in_col",
        "tokens_in_row",
        "text_length",
        "is_digit",
        'font_size',
        'row_dist_from_total',
        "value",
        "rows_in_col",
        "cols_in_row",
        "has_total_below",
    ]
}

X_train, y_train = get_data(TRAIN_FOLDER, config=CONFIG)
X_test, y_test = get_data(TEST_FOLDER, config=CONFIG)

Model = XGB_model(models_path="/home/gleb_siz/ml_training/recipt_parser/models", name="xgb_parser", xgb_paramters={})

Model.train_model(X_train, y_train, X_test, y_test, xgb_param_override={})

print(Model.evaluate_model(X_test, y_test))
