import argparse
import pickle
import pandas as pd

robust_scaler_file = "../model/robust_scaler"
cate_hist_file = "../model/cate_hist"
xgb_model_file = "../model/xgb_model"
input_cols = ['Product Category','Inspected Samples', 'Measurement Samples',
              'Workmanship Samples', 'Number of Item type', 'Number of Styles']
output_col = 'Total Time (H)'

def preprocessCategory(cat_str):
    cat_rst = cat_str.lower()
    cat_array = cat_rst.split(',')
    cat_result = []
    for v in cat_array:
        v = v.strip()
        v = v.rstrip('s')
        if 'boot' in v:
            v = 'boot'
        elif 'crew' in v:
            v = 'crew'
        cat_result.append(v)
    return cat_result[0]


def histogram(data, bins = None):
    if bins == None:
        bins = list(set(data))
    data_dict = {v: 0 for v in bins}
    data_dict['other'] = 0
    for v in data:
        if v in data_dict:
            data_dict[v] += 1
        else:
            data_dict['other'] += 1
    for k in data_dict:
        data_dict[k] = data_dict[k]/len(data)
    return data_dict

def preprocessing(df, cate_hist, rbscaler):
    df['Product Category'] = df['Product Category'].apply(preprocessCategory)
    df['Product Category'] = df['Product Category'].apply(lambda k: cate_hist[k] if k in cate_hist else 0)
    outlier_features = ['Inspected Samples','Workmanship Samples','Measurement Samples',
                  'Number of Item type','Number of Styles']

    df[outlier_features] = rbscaler.transform(df[outlier_features])
    return df[input_cols].values

def predict(df, cate_hist, rbscaler, xgb_model):
    X_test = preprocessing(df, cate_hist, rbscaler)
    return xgb_model.predict(X_test)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='example: python predict.py -i test.csv -rs out.csv')
    parser.add_argument('-i', '--input', help='input file to predict', required=True)
    parser.add_argument('-rs', '--result', help='output file to write the result of testing', required=True)
    args = parser.parse_args()

    input_file = args.input
    result_file = args.result

    df_test = pd.read_csv(input_file)

    # load model
    rbs = pickle.load(open(robust_scaler_file, "rb"))
    cate_hist = pickle.load(open(cate_hist_file, "rb"))
    xgb_model = pickle.load(open(xgb_model_file, "rb"))

    y_test = predict(df_test, cate_hist, rbs, xgb_model)

    df_result = pd.DataFrame(data = {'ID': df_test['ID'], output_col: y_test})
    df_result.to_csv(result_file, index = False)


