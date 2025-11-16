import os
import cv2
import json
import re
import pandas as pd
import numpy as np
# UTILS
# CONSTANTS
# data load
TRAIN_FOLDER = "/home/gleb_siz/ml_training/data/SROIE2019/train"
TEST_FOLDER = "/home/gleb_siz/ml_training/data/SROIE2019/test"
FEATURES = [
## file level
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

total_keywords = [
    "total",
    "sum", 
    "suma", 
    "suma pln",
    "sprzedaz",
]
pattern = re.compile(r'\b(' + '|'.join(total_keywords) + r')\b', re.IGNORECASE)
PRICE_PATTERN = re.compile(
    r'^[A-Za-z $€¥£]*[:=]?\s*\$?\s*[RM]*[+-]?([\d]*[\d.,]+)[ RMDHS]*$'
)
# Combined regex for numeric and short textual dates
date_pattern = re.compile(
    r'('
    r'\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b|'        # YYYY-MM-DD or YYYY/MM/DD
    r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b|'      # DD/MM/YYYY or DD.MM.YY
    r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
    r'January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}\b'
    r')'
)

def match_price(text, pattern=PRICE_PATTERN):
    match = re.search(pattern, text)
    if not match:
        return False
    else:
        return True

def extract_price(text, pattern=PRICE_PATTERN):
    match = re.search(pattern, text)
    if not match:
        return None
    num = match.group(1).replace(',', '.')        # normalize commas
    parts = num.split('.')                        # split by dot
    if len(parts) == 1:
        return float(parts[0])                    # just a plain integer
    integer_part = ''.join(parts[:-1])            # join everything except last
    decimal_part = parts[-1]
    try:
        return float(f"{integer_part}.{decimal_part}")
    except ValueError:
        return None


def load_ocr(path):
    ocr_output_df = pd.DataFrame()
    for f in os.listdir(f"{path}/box"):
        file = f"{path}/box/{f}"
        with open(file) as fl:
            try:
                lines = fl.readlines()
                df = pd.DataFrame(lines, columns=['raw'])
                df['file'] = f
                ocr_output_df = pd.concat([ocr_output_df, df])
            except Exception as e:
                print("Failed to process:", file)
    ocr_output_df[['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']] = ocr_output_df.apply(lambda x: pd.Series(x['raw'].split(',')[:8]), axis=1)
    ocr_output_df['text'] = ocr_output_df.apply(lambda x: ','.join(x['raw'].split(',')[8:]).replace('\n', ''), axis=1)
    ocr_output_df['file'] = ocr_output_df['file'].apply(lambda x: x.split('.')[0])
    ocr_output_df = ocr_output_df.replace('\n', '')
    ocr_output_df = ocr_output_df.dropna()
    for col in ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']:
        ocr_output_df[col] = ocr_output_df[col].astype(int)
    ocr_output_df['x_max'] = ocr_output_df.apply(lambda x: max(x['x1'], x['x2'], x['x3'], x['x4']), axis=1)
    ocr_output_df['y_max'] = ocr_output_df.apply(lambda x: max(x['y1'], x['y2'], x['y3'], x['y4']), axis=1)
    ocr_output_df['x_min'] = ocr_output_df.apply(lambda x: min(x['x1'], x['x2'], x['x3'], x['x4']), axis=1)
    ocr_output_df['y_min'] = ocr_output_df.apply(lambda x: min(x['y1'], x['y2'], x['y3'], x['y4']), axis=1)
    return ocr_output_df


def load_img_data(path):
    img_data = []
    for f in os.listdir(f"{path}/img"):
        file = f"{path}/img/{f}"
        img = cv2.imread(file)
        heigh, width, _ = img.shape
        img_data.append((f, width, heigh))
    img_df = pd.DataFrame(img_data, columns=['file', 'width', 'heigh'])
    img_df['file'] = img_df['file'].apply(lambda x: x.split('.')[0])
    return img_df


def load_entity_data(path):
    labels_df = pd.DataFrame()
    labels = []
    for f in os.listdir(f"{path}/entities"):
        file = f"{path}/entities/{f}"
        with open(file) as fl:
            item = json.loads(fl.read())
            for k, v in item.items():
                labels.append({'label': k, 'text': v, 'file': f.split('.')[0]})
            labels_df = pd.DataFrame(labels)
    return labels_df

def match_labels(train, labels):
    matched_labels = []

    for _, row in train.iterrows():
        file = row['file']
        text = str(row['text'])
        subset = labels[labels['file'] == file]

        matched_label = None
        for _, lab in subset.iterrows():
            lab_text = str(lab['text'])
            # Check substring both ways for robustness
            if lab_text in text or text in lab_text:
                matched_label = lab['label']
                break

        matched_labels.append(matched_label if matched_label else 'other')

    train['label'] = matched_labels
    return train


def add_features(df):
    
    df['width'] = df.groupby(['file'])['x_max'].transform("max")
    df['heigh'] = df.groupby(['file'])['y_max'].transform("max")
    df['file_aspect_ratio'] = df.apply(lambda x: x['width'] / (x['heigh'] + 0.00001), axis=1)
    df['x_max'] = (df['x_max'] / df['width']).round(2)
    df['y_max'] = (df['y_max'] / df['heigh']).round(2)
    df['x_min'] = (df['x_min'] / df['width']).round(2)
    df['y_min'] = (df['y_min'] / df['heigh']).round(2)
    df['token_width'] = df['x_max'] - df['x_min']
    df['token_heigh'] = df['y_max'] - df['y_min']
    df['avg_font'] = df.groupby('file')['token_heigh'].transform("mean")
    df['font_size'] = df['token_heigh'] / df['avg_font']
    df['aspect_ratio'] = df.apply(lambda x: x['token_width'] / (x['token_heigh'] + 0.00001), axis=1)
    df['y_center'] = (df['y_max'] + df['y_min']) / 2
    df['x_center'] = (df['x_max'] + df['x_min']) / 2
    df['x_center_file'] = df.groupby(['file'])['x_center'].transform("mean").round(2)
    df['y_center_file'] = df.groupby(['file'])['y_center'].transform("mean").round(2)
    df['row'] = df['y_center'].round(2)
    df['col'] = df['x_center'].round(1)
    df['row_rank'] = df.groupby('file')['row'].rank(method='dense', ascending=True)
    df['col_rank'] = df.groupby('file')['col'].rank(method='dense', ascending=True)
    df['tokens_in_col'] = df.groupby(['file', 'col'])['text'].transform('count')
    df['tokens_in_row'] = df.groupby(['file', 'row'])['text'].transform('count') 
    df['rows_in_col'] = df.groupby(['file', 'col'])['row_rank'].transform('max')
    df['cols_in_row'] = df.groupby(['file', 'row'])['col_rank'].transform('max') 
    df['has_total_keyword'] = df['text'].apply(lambda t: bool(pattern.search(t)))
    df['has_total_keyword_in_row'] = df.groupby(['file', 'row'])['has_total_keyword'].transform("max")

    totals = df[df['has_total_keyword']][['file', 'row']]
    df = df.merge(totals, on=['file'], how='left', suffixes=('', '_total'))
    df['row_dist_from_total']= np.abs(df['row'] - df['row_total'])
    df = df.sort_values(['file', 'text', 'x_center', 'y_center', 'row_dist_from_total']).groupby(['file', 'text', 'x_center', 'y_center'], as_index=False).first()
    df = df.drop('row_total', axis=1)

    df = df.sort_values(['file', 'row'])
    df['has_total_below'] = (
        df.groupby('file')['has_total_keyword']
        .transform(lambda x: x.iloc[::-1].cummax().iloc[::-1])
    )
    df['has_total_below'] = df.groupby('file')['has_total_below'].shift(-1).fillna(False)

    df['text_length'] = df['text'].apply(lambda x: len(x))
    df['is_digit'] = df['text'].apply(match_price)
    df['value'] = df['text'].apply(extract_price)

    df['contains_date'] = df['text'].apply(lambda x: bool(date_pattern.search(x)))
    df['tokens'] = df.groupby(['file', 'row'])['text'].transform(''.join)
    
    df = df.reset_index()
    
    return df


def fix_total_label(df, threshold=0.015):
    # word total in a row with total label by least diff
    total_keywords = df.copy()
    totals = df.copy()
    total_keywords = total_keywords[total_keywords['has_total_keyword']][['file', 'text', 'has_total_keyword', 'y_center', 'y_max']]
    total_keywords['label'] = 'total'
    totals = totals[totals['label']=='total']

    totals = totals.merge(total_keywords, on=['file'], suffixes=['', '_total'])
    totals['diff_wtotal'] = totals['y_center'] - totals['y_center_total']
    totals = totals.loc[lambda x: x['diff_wtotal'].abs() <= threshold]

    totals = totals.sort_values(['file', 'label', 'y_min'], ascending=[True, True, False])
    totals = totals.groupby(['file', 'label'], as_index=False).first()
    totals = totals[['file', 'text', 'x_min', 'y_min', 'x_max', 'y_max', 'label']]

    df = df.merge(totals, on=['file', 'text', 'x_min', 'y_min', 'x_max', 'y_max',], how='left', suffixes=('', '_true'))
    df['label_true'] = df['label_true'].fillna('other')
    df.loc[lambda x: x['label'] == 'total', 'label'] = df.loc[lambda x: x['label'] == 'total', 'label_true']
    df['label'] = df['label'].apply(lambda x: 1 if x == 'total' else 0)
    
    # ad-hoc fix for own data input
    df.loc[lambda x: (x['file'] == '5807493927290997517')
           & (x['text'] == '"4,58"')
           & (x['y_center'] >= 0.330 ), "label"
           ] = 1

    return df


def plot_labels(df, image_source="/home/gleb_siz/ml_training/data/SROIE2019/train/img"):
    file = f"{image_source}/{df['file'].iloc[0]}.jpg"
    img = cv2.imread(file)
    for i, row in df.iterrows():
        x1, y1, x2, y2 = row.x1, row.y1, row.x3, row.y3
        if row.label ==1:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    img = cv2.resize(img, (800, 1000))
    cv2.imshow("OCR Tokens", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_data(path, config):
    features = config['features']
    ocr_df = load_ocr(path)
    labels_df = load_entity_data(path)
    train_df = match_labels(ocr_df, labels_df)
    features_df = add_features(train_df)
    features_df = fix_total_label(features_df)
    with_total_files = features_df.loc[lambda x: x['label'] == 1]['file'].unique()
    features_df = features_df[features_df['file'].isin(with_total_files)]
    y = features_df['label']
    X = features_df[features]

    return X, y
