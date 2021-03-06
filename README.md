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
    - GridSearchCV と RandomizedSearchCV でハイパーパラメータの調整をした
      - GridSearchCV
        - Best Model Parameter:  {'max_depth': 8, 'max_features': 'log2', 'n_estimators': 10}
        - Train score: 0.5209169050766573
        - Cross Varidation score: 0.4548388971850926
      - RandomizedSearchCV
        - Best Model Parameter:  {'bootstrap': True, 'criterion': 'mse', 'max_depth': 6, 'max_features': 9, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 40}
        - Train score: 0.4946184271369229
        - Cross Varidation score: 0.4550559006191772
  - tutorial2 ; ここまでの成果を使って, submit 用のデータを作成
    - Fare ; Embarked == S のデータを使って, median で埋める
    - Embarked ; 一番多い S で埋める
    - Cabin ; Cabin に特別な特徴が見られなかったので, Cabin の頭文字で Cabin_Cap という特徴量を作り, 一番多い C で埋める
    - Name ; Name から敬称 "Title" を作り, 似たものでグルーピング
    - Age ; Pclass, SibSp, Parch, Sex, Title で RandomForestRegression で推定
    - Survived ; "Pclass", "Sex", "SibSp", "Parch", "Age", "Fare", "Embarked", "Cabin_Cap", "Title" で推定
    - 結果
      - 固定パラメタで RandomForestClassifier ; 0.77990
      - グリッドサーチ ; 0.77511 (何故だ!)
      - グリッドサーチで cv=5 にする ; 0.77990 (なかなか向上しない)
  - tutorial2/tutorial2a.ipynb
    - cross_val_score のテスト
  - tutorial2/tutorial2b.ipynb
    - SelectKBest を使ってパラメタを絞り込んだ. score ; 0.77511
  - tutorial2/tutorial2c.ipynb
    - tutorial2.ipynb から派生
    - tutorial2.ipynb で効果の低かった特徴量 (Cabin, Embarked) を外す
    - cv=3 と cv=5 でトライ → 結果はいずれも 0.77511
  - tutorial2/tutorial2d.ipynb
    - tutorial2c.ipynb から派生
    - Cabin と Embarked は元に戻す
    - 苦し紛れに, "Family_Size" = "SibSp" + "Parch" + 1 を特徴量として設定
    - my_submission2d.csv ; 0.78468 (2839 / 16829)
    - my_submission2d_cv3.csv ; 0.78468 (same as above)
    - my_submission2d_cv5.csv ; 0.78468 (same as above)
    - cv5 において
      - Train score: 0.856341189674523
      - Cross Varidation score: 0.8350072186303434
      - Best Model Parameter:  {'max_depth': 6, 'max_features': 'auto', 'n_estimators': 10}
  - tutorial2/tutorial2d2.ipynb
    - tutorial2d.ipynb から派生. ワンホットエンコーディング後に, 'Sex_female' を drop する
    - my_submission2d2.csv
      - Train score: 0.8383838383838383
    - my_submission2d2_cv3.csv
      - Train score: 0.9001122334455668
      - Cross Varidation score: 0.8361391694725029
      - Best Model Parameter:  {'max_depth': 6, 'max_features': None, 'n_estimators': 300} ← 変わらなかった
      - Score ; 0.77990
    - my_submission2d2_cv5.csv
      - Train score: 0.9001122334455668
      - Cross Varidation score: 0.8338710689849977
      - Best Model Parameter:  {'max_depth': 6, 'max_features': None, 'n_estimators': 300} ; n_estimators が変わったが, これは cv=3 と同じ
      - Score ; 0.77990
  - tutorial2/tutorial2e.ipynb
    - tutorial2d.ipynb からの派生. アルゴリズムを RandomForestClassifier から SVM に変えた
    - my_submission2e.csv ; 0.76076
    - my_submission2e_cv3 ; 0.76076
  - tutorial2/tutorial2f.ipynb
    - tutorial2d.ipynb の派生
    - 影響の小さい特徴量を drop する
      - ["Cabin_Cap_A", "Cabin_Cap_G", "Cabin_Cap_T", "Cabin_Cap_F"]
        - 0.77511 だった
      - ["Cabin_Cap_A", "Cabin_Cap_G", "Cabin_Cap_T", "Cabin_Cap_F", "Cabin_Cap_B", "Embarked_Q"]
        - 0.77511 だった
        - 各特徴量の影響度のグラフが大きく変わった
  - tutorial2/tutorial2g.ipynb
    - tutorial2d.ipynb の派生. RandomizedSearchCV でハイパーパラメータを探索
    - Best Model Parameter:  {'bootstrap': False, 'criterion': 'entropy', 'max_depth': None, 'max_features': 10, 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 10}
    - Train score: 0.898989898989899
    - Cross Varidation score: 0.8406377502981608
    - my_submission_grid2g_cv5.csv ; 0.78468 (same as "d")
  - tutorial2/tutorial2g2.ipynb
    - tutorial2g2.ipynb の派生. "Age" の欠損値の推定を, RandomizedSearchCV で得られたハイパーパラメタを使用
    - Best Model Parameter:  {'bootstrap': True, 'criterion': 'gini', 'max_depth': 10, 'max_features': 10, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 240}
    - Train score: 0.9102132435465768
    - Cross Varidation score: 0.8406189190885694
    - CV スコアは, 1 つ前のほうがちょっと高め
    - 0.78229 (下がった)
  - tutorial2/tutorial2g3.ipynb
    - tutorial2g.ipynb の派生。tutorial2g.ipynb の結果をもとに，もう 1 度 RandomizedSearchCV をして，ハイパーパラメタを改善
    - Best Model Parameter:  {'n_estimators': 15, 'min_samples_split': 12, 'min_samples_leaf': 6, 'max_features': 12, 'max_depth': 12, 'criterion': 'gini', 'bootstrap': False}
    - Train score: 0.898989898989899
    - Cross Varidation score: 0.8473667691921412
    - my_submission_grid2g3_cv5.csv ; 0.78468 (same as "g" and "d")
  - tutorial2/tutorial2g4.ipynb
    - tutorial2g3.ipynb で得られたハイパーパラメータを用い，特徴量 k=18 で推定
    - test score ; 0.8372795179210344
    - score ; 0.7790
  - tutorial2/tutorial2g5.ipynb
    - tutorial2g3.ipynb の派生。"Sex_female" を drop して，RandomizeSearchCV をやり直す
    - Best Model Parameter:  {'n_estimators': 10, 'min_samples_split': 8, 'min_samples_leaf': 6, 'max_features': 14, 'max_depth': None, 'criterion': 'gini', 'bootstrap': False}
    - Train score: 0.8956228956228957
    - Cross Varidation score: 0.8451195781809051

| File Name | Train score | CV score | Score |
| ---- | ---- | ---- | ---- |
| tutorial2d (cv=5) | 0.856341189674523 | 0.8350072186303434 | 0.78468 |
| tutorial2d2 (cv=5) | 0.9001122334455668 | 0.8338710689849977 | 0.77990 |
| tutorial2g (cv=5) | 0.898989898989899 | 0.8406377502981608 | 0.78468 |
| tutorial2g2 (cv=5) | 0.9102132435465768 | 0.8406189190885694 | 0.78229 |
| tutorial2g3 (cv=5) | 0.898989898989899 | 0.8473667691921412 | 0.78468 |
| tutorial2g4 (k=18) |  | 0.8372795179210344 | 0.77990 |
| tutorial3g5 (cv=5) | 0.8956228956228957 | 0.8451195781809051 | 0.77990 |

- 参考
  - 1 / 418 = 0.0023923

- scripts
  - show_data.py ; draw some graphs

- graphs
  - sex.svg ; sex
  - age.svg ; age
