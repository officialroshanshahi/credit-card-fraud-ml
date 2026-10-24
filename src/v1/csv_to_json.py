import pandas as pd
from src.v1.preprocessing import load_data

_, x_test, _, y_test = load_data()

y_test_df = pd.DataFrame(y_test, columns=['Class'], index=x_test.index).astype(int)

files_1 = 'class_1_test.json'
files_0 = 'class_0_test.json'

_test = pd.concat([x_test, y_test_df], axis=1)

_test_1 = _test[_test['Class'] == 1]
_test_0 = _test[_test['Class'] == 0]

_test_1.to_json(files_1, orient='records', indent=4)
_test_0.to_json(files_0, orient='records', indent=4)