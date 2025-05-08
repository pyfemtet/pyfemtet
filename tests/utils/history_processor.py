import csv
import json
from pyfemtet._i18n import ENCODING
from pyfemtet.opt.history._history import ColumnManager, History


def remove_additional_data(history_path):

    # csv を読み込む
    rows = []
    with open(history_path, 'r', encoding=ENCODING) as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    # trial column のあるところの metadata を削除
    # trial column を get
    idx = rows[2].index(ColumnManager._get_additional_data_column())
    rows[0][idx] = json.dumps(dict())

    # 書き戻す
    new_path = history_path
    with open(new_path, 'w', encoding=ENCODING, newline='\n') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # history として正常か確認
    history = History()
    history.load_csv(new_path, True)
    # history.get_df()[history.prm_names + history.obj_names]
    # history._records.column_manager.meta_columns
    assert history.additional_data == dict()
