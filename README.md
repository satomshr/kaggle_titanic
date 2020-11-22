# kaggle_titanic
Codes for Kaggle/Titanic

- folders
  - tutorial ; see details of tutorial scripts
  - tutorial1 ; study the effects of parameters of RandomForestClassifier
  - features0
    - features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare", "Embarked"]
    - RandomForestClassifier ; {'max_depth': 6, 'max_features': None, 'n_estimators': 30}
    - Score ; 0.77990 <- 0.77751
  - features_title ; check title of "Name" feature
    - feature_title.py ; 「敬称」を分析したスクリプト
    - feature_title_mk_submission.py
      - 上記のスクリプトを, submit 用のデータを作る用に改造
      - 「敬称」を考慮したら, 0.76794 に下がってしまった
      - 影響度を調べたら "Embarked" の影響が小さかったので, それを外してみたが変わらなかった
      - さらに "SibSp" と "Parch" を外したけど, ダメだった
  - features_ticket ; check "Ticket"
  - features_cabin ; check "Cabin"
    - feature_cabin.py ; "Cabin" の記号ごとに生存率を調査. あまりパっとしなかった
    - feature_cabin_cap.py ; "Cabin" の頭文字でグルーピング. 生存率の高い記号はあるが, あまりメリハリは無い
  - features_age ; check "Age"
    - "Age" の欠損値の補完方法の検討. "Pclass", "SibSp", "Parch", "Sex" に, 名前から作った敬称 "Title" を加えることで, "Age" の予測精度が上がった
  - tutorial2 ; ここまでの成果を使って, submit 用のデータを作成
    - Fare ; Embarked == S のデータを使って, median で埋める
    - Embarked ; 一番多い S で埋める
    - Cabin ; Cabin に特別な特徴が見られなかったので, Cabin の頭文字で Cabin_Cap という特徴量を作り, 一番多い C で埋める
    - Name ; Name から敬称 "Title" を作り, 似たものでグルーピング
    - Age ; Pclass, SibSp, Parch, Sex, Title で RandomForestRegression で推定
    - Survived ; "Pclass", "Sex", "SibSp", "Parch", "Age", "Fare", "Embarked", "Cabin_Cap", "Title" で推定

- scripts
  - show_data.py ; draw some graphs

- graphs
  - sex.svg ; sex
  - age.svg ; age
