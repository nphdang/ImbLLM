import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def normalize_data(X):
    # normalize X
    normalizer = MinMaxScaler().fit(X)

    return normalizer

def encode_data(X_df, y_df, numerical_var_names, categorical_var_names, encoding="ordinal"):
    # print("encoding: {}".format(encoding))
    batch_size_predict = 64
    # encode data
    if encoding == "original":
        return X_df, y_df

    if encoding == "ordinal":  # encode each categorical value into an integer
        X = np.array(X_df.loc[:, numerical_var_names])
        # convert categorical values to categorical indices
        for categorical_var_name in categorical_var_names:
            categorical_var = np.array(pd.Categorical(X_df.loc[:, categorical_var_name]))
            # categorical_val = np.unique(categorical_var)
            categorical_val = list(pd.Categorical(X_df.loc[:, categorical_var_name]).categories)
            categorical_dict = [{'value': c, 'index': i} for i, c in enumerate(categorical_val)]
            categorical_names = [categorical_dict[idx]['value'] for idx in range(len(categorical_val))]
            categorical_indices = [categorical_dict[idx]['index'] for idx in range(len(categorical_val))]
            for idx in range(len(categorical_val)):
                categorical_var[categorical_var == categorical_names[idx]] = categorical_indices[idx]
            X = np.append(X, categorical_var.reshape(-1, 1), axis=1)
    elif encoding == "onehot":  # encode each categorical value into a one-hot vector
        X = X_df.loc[:, numerical_var_names]
        # convert categorical values to discrete values
        for categorical_var_name in categorical_var_names:
            categorical_var = pd.Categorical(X_df.loc[:, categorical_var_name])
            # set one dummy variable if it's boolean
            if len(categorical_var.categories) == 2:
                drop_first = True
            else:
                drop_first = False
            dummies = pd.get_dummies(categorical_var, prefix=categorical_var_name, drop_first=drop_first)
            X = pd.concat([X, dummies], axis=1)
        X = X.to_numpy()    

    # convert labels to integer array
    y = np.array(y_df)
    y = np.array([int(val) for val in y])

    return X, y

def gen_train_test_data(dataset="", train_size=1.0, test_size=0.2, seed=123):
    print("dataset: {}, seed: {}".format(dataset, seed))
    print("train_size: {}, test_size: {}".format(train_size, test_size))

    numerical_var_names, categorical_var_names, class_var_name = None, None, None
    # load raw data
    df = pd.read_csv("./data/{}.csv".format(dataset), header=0, sep=",")
    if dataset == "adult":
        numerical_var_names = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        categorical_var_names = ["race", "sex", "workclass", "marital-status", "occupation", "relationship"]
        class_var_name = "income-per-year"
    if dataset == "german":
        numerical_var_names = ["month", "credit_amount", "investment_as_income_percentage",
                               "residence_since", "number_of_credits", "people_liable_for"]
        categorical_var_names = ["age", "sex", "status", "credit_history", "purpose",
                                 "savings", "employment", "other_debtors", "property",
                                 "installment_plans", "housing", "skill_level", "telephone",
                                 "foreign_worker"]
        class_var_name = "credit"
    if dataset == "bank":
        numerical_var_names = ["balance", "duration", "campaign", "pdays", "previous"]
        categorical_var_names = ["age", "marital", "job", "education", "default", "housing", "loan", "contact", "poutcome"]
        class_var_name = "subscribe"
    if dataset == "car":
        numerical_var_names = ["doors", "persons"]
        categorical_var_names = ["buying", "maint", "lug_boot", "safety"]
        class_var_name = "target"
    if dataset == "sick":
        numerical_var_names = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
        categorical_var_names = ["sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
                                 "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
                                 "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
                                 "TSH_measured", "T3_measured", "TT4_measured", "T4U_measured", "FTI_measured",
                                 "referral_source"]
        class_var_name = "Class"
    if dataset == "credit_card":
        numerical_var_names = ["Customer_Age", "Dependent_count", "Months_on_book", "Total_Relationship_Count",
                               "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
                               "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
                               "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"]
        categorical_var_names = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
        class_var_name = "Attrition_Flag"
    if dataset == "fuel":
        numerical_var_names = ["ENGINE_SIZE", "CYLINDERS", "COEMISSIONS"]
        categorical_var_names = ["MAKE", "MODEL", "VEHICLE_CLASS", "TRANSMISSION", "FUEL"]
        class_var_name = "CONSUMPTION"
        df = df.dropna(how="any")
        # plt.hist(df[class_var_name])
        # plt.show()
        cut_off = 15
        high_indices = np.where(df[class_var_name] >= cut_off)[0]
        low_indices = np.where(df[class_var_name] < cut_off)[0]
        df.iloc[high_indices, -1] = "high"
        df.iloc[low_indices, -1] = "low"
    if dataset == "gem_price":
        numerical_var_names = ["carat", "depth", "table", "x", "y", "z"]
        categorical_var_names = ["cut", "color", "clarity"]
        class_var_name = "price"
        df = df.dropna(how="any")
        # plt.hist(df[class_var_name])
        # plt.show()
        cut_off = 2500
        high_indices = np.where(df[class_var_name] >= cut_off)[0]
        low_indices = np.where(df[class_var_name] < cut_off)[0]
        df.iloc[high_indices, -1] = "high"
        df.iloc[low_indices, -1] = "low"
    if dataset == "housing":
        numerical_var_names = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                               "population", "households", "median_income"]
        categorical_var_names = ["ocean_proximity"]
        class_var_name = "median_house_value"
        df = df.dropna(how="any")
        # plt.hist(df[class_var_name])
        # plt.show()
        cut_off = 200000
        high_indices = np.where(df[class_var_name] >= cut_off)[0]
        low_indices = np.where(df[class_var_name] < cut_off)[0]
        df.iloc[high_indices, -1] = "high"
        df.iloc[low_indices, -1] = "low"
    if dataset == "insurance":
        numerical_var_names = ["age", "bmi", "children"]
        categorical_var_names = ["sex", "smoker", "region"]
        class_var_name = "charges"
        df = df.dropna(how="any")
        # plt.hist(df[class_var_name])
        # plt.show()
        cut_off = 10000
        high_indices = np.where(df[class_var_name] >= cut_off)[0]
        low_indices = np.where(df[class_var_name] < cut_off)[0]
        df.iloc[high_indices, -1] = "high"
        df.iloc[low_indices, -1] = "low"    

    # get feature names
    feature_names = np.append(numerical_var_names, categorical_var_names)
    X_df, y_df = df[feature_names], df[class_var_name]
    # create test set
    X_train_, X_test, y_train_, y_test = train_test_split(X_df, y_df, test_size=test_size,
                                                          shuffle=True, stratify=y_df, random_state=seed)
    # create training set
    if train_size == 1.0:
        X_train = X_train_
        y_train = y_train_
    else:
        X_train, _, y_train, _ = train_test_split(X_train_, y_train_, test_size=(1.0 - train_size),
                                                  shuffle=True, stratify=y_train_, random_state=seed)
    X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    X_test, y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    # convert labels to indices
    labels_train = pd.Categorical(y_train)
    indices_train = labels_train.codes
    for idx in range(len(y_train)):
        y_train[idx] = indices_train[idx]
    y_train = y_train.astype('int8')
    y_train_dist = np.bincount(y_train)
    print("dataset: {}, imbalance ratio: {}".format(dataset, round(max( y_train_dist) / min( y_train_dist), 1)))
    labels_test = pd.Categorical(y_test)
    indices_test = labels_test.codes
    for idx in range(len(y_test)):
        y_test[idx] = indices_test[idx]
    y_test = y_test.astype('int8')
    
    n_train, n_test, n_feature, n_class = X_train.shape[0], X_test.shape[0], len(feature_names), len(np.unique(y_train))
    print("X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))
    print("X_test: {}, y_test: {}".format(X_test.shape, y_test.shape))
    print("n_train: {}, n_test: {}, n_feature: {}, n_class: {}".format(n_train, n_test, n_feature, n_class))
    print("feature_names: {}, class_var_name: {}".format(feature_names, class_var_name))
    print("numerical_var_names: {}, categorical_var_names: {}".format(numerical_var_names, categorical_var_names))

    return X_train, y_train, X_test, y_test, n_train, n_test, n_feature, n_class, \
           feature_names, numerical_var_names, categorical_var_names, class_var_name

