import pandas as pd
from IPython.display import display
from sklearn.model_selection import GroupKFold, StratifiedKFold
from config import global_params


def make_folds(train_csv: pd.DataFrame, config: global_params.MakeFolds()) -> pd.DataFrame:
    """Split the given dataframe into training folds."""
    # TODO: add options for cv_scheme as it is cumbersome here.
    if config.cv_schema == "StratifiedKFold":
        df_folds = train_csv.copy()
        skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X=df_folds[config.image_col_name], y=df_folds[config.class_col_name])
        ):
            df_folds.loc[val_idx, "fold"] = int(fold + 1)
        df_folds["fold"] = df_folds["fold"].astype(int)
        print(df_folds.groupby(["fold", config.class_col_name]).size())

    elif config.cv_schema == "GroupKfold":
        df_folds = train_csv.copy()
        gkf = GroupKFold(n_splits=config.num_folds)
        groups = df_folds[config.group_kfold_split].values
        for fold, (train_index, val_index) in enumerate(
            gkf.split(X=df_folds, y=df_folds[config.class_col_name], groups=groups)
        ):
            df_folds.loc[val_index, "fold"] = int(fold + 1)
        df_folds["fold"] = df_folds["fold"].astype(int)
        try:
            print(df_folds.groupby(["fold", config.class_col_name]).size())
        except:
            display(df_folds)

    else:  # No CV Schema used in this file, but custom one
        df_folds = train_csv.copy()
        try:
            print(df_folds.groupby(["fold", config.class_col_name]).size())
        except:
            display(df_folds)

    df_folds.to_csv(config.folds_csv, index=False)

    return df_folds
