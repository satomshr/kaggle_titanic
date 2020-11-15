# kaggle_titanic
Codes for Kaggle/Titanic

- folders
  - tutorial ; see details of tutorial scripts
  - tutorial1 ; study the effects of parameters of RandomForestClassifier
  - features1
    - features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare", "Embarked"]
    - RandomForestClassifier ; {'max_depth': 6, 'max_features': None, 'n_estimators': 30}
    - Score ; 0.77990 <- 0.77751
  - features_title ; check title of "Name" feature
    - feature_title.py ; 「敬称」を分析したスクリプト
    - feature_title_mk_submission.py
      - 上記のスクリプトを, submit 用のデータを作る用に改造
      - 「敬称」を考慮したら, 0.76794 に下がってしまった
      - 影響度を調べたら "Embarked" の影響が小さかったので, それを外してみたが変わらなかった

- scripts
  - show_data.py ; draw some graphs
  
- graphs
  - sex.svg ; sex
  - age.svg ; age
