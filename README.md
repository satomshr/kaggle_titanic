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

- scripts
  - show_data.py ; draw some graphs

- graphs
  - sex.svg ; sex
  - age.svg ; age
