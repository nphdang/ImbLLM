import copy
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, SMOTENC
from sdv.tabular import TVAE, CTGAN
from imblearn.datasets import make_imbalance
from tabular_llm.great import GReaT
from tabular_llm.taptap import Taptap
from tabular_llm.predllm import PredLLM
from tabular_llm.imbllm import ImbLLM
from tabular_llm.great_utils import _encode_row_partial
import read_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="fuel", type=str, nargs='?', help='dataset name')
parser.add_argument('--trainsize', default="1.0", type=str, nargs='?', help='size of training set')
parser.add_argument('--testsize', default="0.2", type=str, nargs='?', help='size of test set')
parser.add_argument('--g_encode', default="original", type=str, nargs='?', help='generation encoding')
parser.add_argument('--c_encode', default="ordinal", type=str, nargs='?', help='classification encoding')
parser.add_argument('--imbalance', default="imb_llm", type=str, nargs='?', help='imbalance method')
parser.add_argument('--ratio', default="0.2", type=str, nargs='?', help='imbalance ratio')
parser.add_argument('--sm_ratio', default="-1", type=str, nargs='?', help='smote ratio to train LLM')
parser.add_argument('--runs', default="3", type=str, nargs='?', help='no of times to run algorithm')
parser.add_argument('--location', default="server", type=str, nargs='?', help='location to run algorithm')
args = parser.parse_args()

dataset_input = args.dataset
train_size = float(args.trainsize)
test_size = float(args.testsize)
g_encoding = args.g_encode
c_encoding = args.c_encode
imbalance_method = args.imbalance
imbalance_ratio = float(args.ratio)
smote_ratio = float(args.sm_ratio)
n_run = int(args.runs)
location = args.location

llm_batch_size = 32
llm_epochs = 50

if dataset_input == "all":
    datasets = ["german", "fuel", "insurance", "housing", "bank", "adult", "gem_price", "car", "sick", "credit_card"]
else:
    datasets = [dataset_input]
print("datasets: {}".format(datasets))

def ratio_func(y, imb_ratio, minority_class):
    target_stats = Counter(y)
    return {minority_class: int(imb_ratio * target_stats[minority_class])}

for dataset in datasets:
    print("dataset: {}".format(dataset))
    acc_xgb_run, f1_xgb_run, auc_xgb_run = np.zeros(n_run), np.zeros(n_run), np.zeros(n_run)
    for run in range(n_run):
        print("run: {}".format(run))
        np.random.seed(run)
        # load data
        X_train, y_train, X_test, y_test, n_train, n_test, _, n_class, \
        feature_names, numerical_names, categorical_names, target_name = \
            read_data.gen_train_test_data(dataset, train_size, test_size, seed=run)
        class_dist = np.bincount(y_train)
        print("balanced train class_dist: {}".format(class_dist))
        print("balanced test class_dist: {}".format(np.bincount(y_test)))

        # make imbalance
        minor_class = np.argmin(class_dist)
        X_imb, y_imb = make_imbalance(X_train, y_train, random_state=run,
                                      sampling_strategy=ratio_func,
                                      **{"imb_ratio": imbalance_ratio, "minority_class": minor_class})
        class_dist = np.bincount(y_imb)
        print("imbalanced train class_dist: {}".format(class_dist))
        X_imb, y_imb = X_imb.reset_index(drop=True), y_imb.reset_index(drop=True)

        major_sample = np.max(class_dist)
        minor_sample = np.min(class_dist)
        gen_sample = major_sample - minor_sample
        print("minor_class: {}, minor_sample: {}, gen_sample: {}".format(minor_class, minor_sample, gen_sample))

        # encode balanced and imbalanced train dataset
        # standard and SMOTE methods use "ordinal", deep learning and LLMs methods use "original"
        X_bal_enc, y_bal_enc = read_data.encode_data(X_train, y_train,
                                                     numerical_names, categorical_names, encoding=g_encoding)
        X_imb_enc, y_imb_enc = read_data.encode_data(X_imb, y_imb,
                                                     numerical_names, categorical_names, encoding=g_encoding)
        # handle imbalance
        min_sample_per_class = 5
        if imbalance_method == "balance":
            X_train, y_train = X_bal_enc, y_bal_enc
        if imbalance_method == "standard":
            X_train, y_train = X_imb_enc, y_imb_enc
        if imbalance_method == "smote":
            if minor_sample <= min_sample_per_class:
                sm = RandomOverSampler(random_state=run)
            else:
                sm = SMOTE(random_state=run)
            X_train, y_train = sm.fit_resample(X_imb_enc, y_imb_enc)
            class_dist = np.bincount(y_train)
            print("SMOTE class_dist: {}".format(class_dist))
        if imbalance_method == "smotenc":
            n_feature = len(feature_names)
            n_continous = len(numerical_names)
            n_categorical = len(categorical_names)
            categorical_indices = [idx for idx in range(n_continous, n_feature)]
            assert n_categorical == len(categorical_indices)
            if minor_sample <= min_sample_per_class:
                smnc = RandomOverSampler(random_state=run)
            else:
                smnc = SMOTENC(random_state=run, categorical_features=categorical_indices)
            X_train, y_train = smnc.fit_resample(X_imb_enc, y_imb_enc)
            class_dist = np.bincount(y_train)
            print("SMOTE_NC class_dist: {}".format(class_dist))
        if imbalance_method == "adasyn":
            if minor_sample <= min_sample_per_class:
                ada = RandomOverSampler(random_state=run)
            else:
                ada = ADASYN(random_state=run)
            X_train, y_train = ada.fit_resample(X_imb_enc, y_imb_enc)
            class_dist = np.bincount(y_train)
            print("ADASYN class_dist: {}".format(class_dist))
        if imbalance_method == "ctgan":
            n_feature = X_imb_enc.shape[1]
            minor_indices = np.where(y_imb_enc == minor_class)[0]
            if g_encoding == "ordinal":
                X_minor, y_minor = X_imb_enc[minor_indices], y_imb_enc[minor_indices]
                data_ctgan_train = np.append(X_minor, y_minor.reshape(-1, 1), axis=1)
                data_ctgan_train = pd.DataFrame(data_ctgan_train)
                data_ctgan_train = data_ctgan_train.add_prefix("col_")
            elif g_encoding == "original":
                X_minor, y_minor = X_imb_enc.iloc[minor_indices, :], y_imb_enc[minor_indices]
                X_minor, y_minor = X_minor.reset_index(drop=True), y_minor.reset_index(drop=True)
                data_ctgan_train = copy.deepcopy(X_minor)
                data_ctgan_train[target_name] = y_minor
            ct_gan = CTGAN()
            ct_gan.fit(data_ctgan_train)
            data_ctgan_new = ct_gan.sample(num_rows=gen_sample,
                                           output_file_path="_{}_{}_{}_{}_{}.csv".
                                           format(dataset, g_encoding, imbalance_method, imbalance_ratio, run))
            if g_encoding == "ordinal":
                X_minor_new = data_ctgan_new.iloc[:, :-1].to_numpy().reshape(-1, n_feature)
                y_minor_new = data_ctgan_new.iloc[:, -1].to_numpy(dtype=int).reshape(-1, )
                X_train, y_train = np.append(X_imb_enc, X_minor_new, axis=0), np.append(y_imb_enc, y_minor_new, axis=0)
            elif g_encoding == "original":
                X_minor_new = data_ctgan_new.iloc[:, :-1]
                y_minor_new = data_ctgan_new.iloc[:, -1]
                X_bal_df = pd.concat([X_imb_enc, X_minor_new], axis=0, ignore_index=True)
                y_bal_df = pd.concat([y_imb_enc, y_minor_new], axis=0, ignore_index=True)
                X_train, y_train = read_data.encode_data(X_bal_df, y_bal_df,
                                                         numerical_names, categorical_names, encoding=c_encoding)
            class_dist = np.bincount(y_train)
            print("CTGAN class_dist: {}".format(class_dist))
        if imbalance_method == "tvae":
            n_feature = X_imb_enc.shape[1]
            minor_indices = np.where(y_imb_enc == minor_class)[0]
            if g_encoding == "ordinal":
                X_minor, y_minor = X_imb_enc[minor_indices], y_imb_enc[minor_indices]
                data_tvae_train = np.append(X_minor, y_minor.reshape(-1, 1), axis=1)
                data_tvae_train = pd.DataFrame(data_tvae_train)
                data_tvae_train = data_tvae_train.add_prefix("col_")
            elif g_encoding == "original":
                X_minor, y_minor = X_imb_enc.iloc[minor_indices, :], y_imb_enc[minor_indices]
                X_minor, y_minor = X_minor.reset_index(drop=True), y_minor.reset_index(drop=True)
                data_tvae_train = copy.deepcopy(X_minor)
                data_tvae_train[target_name] = y_minor
            t_vae = TVAE()
            t_vae.fit(data_tvae_train)
            data_tvae_new = t_vae.sample(num_rows=gen_sample,
                                         output_file_path="_{}_{}_{}_{}_{}.csv".
                                         format(dataset, g_encoding, imbalance_method, imbalance_ratio, run))
            if g_encoding == "ordinal":
                X_minor_new = data_tvae_new.iloc[:, :-1].to_numpy().reshape(-1, n_feature)
                y_minor_new = data_tvae_new.iloc[:, -1].to_numpy(dtype=int).reshape(-1, )
                X_train, y_train = np.append(X_imb_enc, X_minor_new, axis=0), np.append(y_imb_enc, y_minor_new, axis=0)
            elif g_encoding == "original":
                X_minor_new = data_tvae_new.iloc[:, :-1]
                y_minor_new = data_tvae_new.iloc[:, -1]
                X_bal_df = pd.concat([X_imb_enc, X_minor_new], axis=0, ignore_index=True)
                y_bal_df = pd.concat([y_imb_enc, y_minor_new], axis=0, ignore_index=True)
                X_train, y_train = read_data.encode_data(X_bal_df, y_bal_df,
                                                         numerical_names, categorical_names, encoding=c_encoding)
            class_dist = np.bincount(y_train)
            print("TVAE class_dist: {}".format(class_dist))
        if imbalance_method == "great":
            X_y_imb_df = copy.deepcopy(X_imb_enc)
            X_y_imb_df[target_name] = y_imb_enc
            great = GReaT(llm='distilgpt2', batch_size=llm_batch_size, epochs=llm_epochs)
            great.fit(data=X_y_imb_df, numeric_cols=numerical_names, categorical_cols=categorical_names)
            # generate data conditioned on the minority label
            encoded_minority = "{} is {},".format(target_name, minor_class)
            tokenized_minority = great.tokenizer(encoded_minority)["input_ids"]
            X_y_minor_new = great.sample_imb(n_samples=gen_sample,
                                             minority_cond=tokenized_minority, minority_class=minor_class)
            X_y_bal_df = pd.concat([X_y_imb_df, X_y_minor_new], axis=0, ignore_index=True)
            # encode generated tabular data using "ordinal"
            X_train, y_train = read_data.encode_data(X_y_bal_df.iloc[:, :-1], X_y_bal_df.iloc[:, -1],
                                                     numerical_names, categorical_names, encoding=c_encoding)
            class_dist = np.bincount(y_train)
            print("Great class_dist: {}".format(class_dist))
        if imbalance_method == "taptap":
            X_y_imb_df = copy.deepcopy(X_imb_enc)
            X_y_imb_df[target_name] = y_imb_enc
            taptap = Taptap(llm='ztphs980/taptap-distill',
                            experiment_dir='./experiment_taptap/',
                            batch_size=llm_batch_size, epochs=llm_epochs,
                            numerical_modeling='split', gradient_accumulation_steps=2)
            taptap.fit(X_y_imb_df, target_col=target_name, task="classification",
                       numeric_cols=numerical_names, categorical_cols=categorical_names)
            # generate data conditioned on the minority label
            encoded_minority = "{} is {},".format(target_name, minor_class)
            tokenized_minority = taptap.tokenizer(encoded_minority)["input_ids"]
            X_y_minor_new = taptap.sample_imb(n_samples=gen_sample, minority_cond=tokenized_minority,
                                              data=X_y_imb_df, task="classification", max_length=1024)

            # predict labels for minority synthetic samples using an external classifier
            X_minor_new, y_minor_new = X_y_minor_new.iloc[:, :-1], X_y_minor_new.iloc[:, -1]
            X_minor_new_c_enc, y_minor_new_c_enc = read_data.encode_data(X_minor_new, y_minor_new,
                                                                         numerical_names, categorical_names,
                                                                         encoding=c_encoding)
            X_imb_c_enc, y_imb_c_enc = read_data.encode_data(X_imb, y_imb,
                                                             numerical_names, categorical_names,
                                                             encoding=c_encoding)
            xgb = XGBClassifier(random_state=run)
            xgb.fit(X_imb_c_enc, y_imb_c_enc)
            y_minor_new = xgb.predict(X_minor_new_c_enc)
            y_minor_new = pd.Series(y_minor_new)

            X_bal_df = pd.concat([X_imb_enc, X_minor_new], axis=0, ignore_index=True)
            y_bal_df = pd.concat([y_imb_enc, y_minor_new], axis=0, ignore_index=True)
            X_train, y_train = read_data.encode_data(X_bal_df, y_bal_df,
                                                     numerical_names, categorical_names, encoding=c_encoding)
            class_dist = np.bincount(y_train)
            print("TapTap class_dist: {}".format(class_dist))
        if imbalance_method == "pred_llm":
            X_y_imb_df = copy.deepcopy(X_imb_enc)
            X_y_imb_df[target_name] = y_imb_enc
            predllm = PredLLM(llm='distilgpt2', batch_size=llm_batch_size, epochs=llm_epochs)
            predllm.fit(data=X_y_imb_df, numeric_cols=numerical_names, categorical_cols=categorical_names)
            # compute length of input sequence
            prompt_lens = []
            for i in range(X_y_imb_df.shape[0]):
                encoded_text = _encode_row_partial(X_y_imb_df.iloc[i], shuffle=False)
                prompt_len = len(predllm.tokenizer(encoded_text)["input_ids"])
                prompt_lens.append(prompt_len)
            prompt_lens = np.array(prompt_lens)
            max_prompt_len = np.max(prompt_lens)
            X_y_minor_new = predllm.sample_imb(n_samples=gen_sample, minority_class=minor_class,
                                               max_length=max_prompt_len, task="classification")
            X_y_bal_df = pd.concat([X_y_imb_df, X_y_minor_new], axis=0, ignore_index=True)
            # encode generated tabular data using "ordinal"
            X_train, y_train = read_data.encode_data(X_y_bal_df.iloc[:, :-1], X_y_bal_df.iloc[:, -1],
                                                     numerical_names, categorical_names, encoding=c_encoding)
            class_dist = np.bincount(y_train)
            print("Pred_LLM class_dist: {}".format(class_dist))
        if imbalance_method == "imb_llm":
            # interpolate minority samples using SMOTE-NC
            n_feature, n_continuous, n_categorical = len(feature_names), len(numerical_names), len(categorical_names)
            continuous_indices = [idx for idx in range(0, n_continuous)]
            categorical_indices = [idx for idx in range(n_continuous, n_feature)]
            # encode imbalanced data using ordinal
            X_imb_ord, y_imb_ord = read_data.encode_data(X_imb, y_imb,
                                                         numerical_names, categorical_names, encoding="ordinal")
            # balance data
            if minor_sample <= min_sample_per_class:
                smnc = RandomOverSampler(random_state=run)
            else:
                smnc = SMOTENC(random_state=run, categorical_features=categorical_indices)
            X_bal_ord, y_bal_ord = smnc.fit_resample(X_imb_ord, y_imb_ord)
            # obtain SMOTE minority samples
            n_imb, n_rebal = X_imb_ord.shape[0], X_bal_ord.shape[0]
            sm_indices = [idx for idx in range(n_imb, n_rebal)]
            X_sm, y_sm = X_bal_ord[sm_indices], y_bal_ord[sm_indices]
            # round up continuous features
            for idx in continuous_indices:
                X_sm[:, idx] = np.around(X_sm[:, idx].astype("float"), 2)
            X_y_sm_df = np.append(X_sm, y_sm.reshape(-1, 1), axis=1)
            X_y_sm_df = pd.DataFrame(X_y_sm_df)
            X_y_sm_df.columns = np.append(feature_names, target_name)

            # obtain minority samples to train LLM
            minor_indices = np.where(y_imb_enc == minor_class)[0]
            X_minor, y_minor = X_imb_enc.iloc[minor_indices, :], y_imb_enc.iloc[minor_indices]
            X_minor, y_minor = X_minor.reset_index(drop=True), y_minor.reset_index(drop=True)
            X_y_imb_df = copy.deepcopy(X_minor)
            X_y_imb_df[target_name] = y_minor

            # select how many SMOTE minority samples to train LLM
            if smote_ratio == -1:
                sm_sample = gen_sample
            else:
                sm_sample = int(smote_ratio * gen_sample)
            print("#minority: {}, #smote: {}, #sm_to_add: {}".format(minor_sample, gen_sample, sm_sample))
            X_y_sm_df = X_y_sm_df.iloc[:sm_sample, :]
            print("X_y_sm_df: {}, X_y_imb_df: {}".format(X_y_sm_df.shape, X_y_imb_df.shape))
            # convert categorical columns of SMOTE from indices to categories
            # as X_y_imb_df is smaller than X_y_sm_df, NA values will happen
            for feature in categorical_names:
                X_y_sm_df[feature] = X_y_imb_df[feature]

            imbllm = ImbLLM(llm='distilgpt2', batch_size=llm_batch_size, epochs=llm_epochs)
            imbllm.fit(data=X_y_imb_df, cont_data=X_y_sm_df,
                       numeric_cols=numerical_names, categorical_cols=categorical_names)
            # generate data conditioned on the minority label
            encoded_minority = "{} is {},".format(target_name, minor_class)
            tokenized_minority = imbllm.tokenizer(encoded_minority)["input_ids"]
            # compute length of input sequence
            prompt_lens = []
            for i in range(X_y_imb_df.shape[0]):
                encoded_text = _encode_row_partial(X_y_imb_df.iloc[i], shuffle=False)
                prompt_len = len(imbllm.tokenizer(encoded_text)["input_ids"])
                prompt_lens.append(prompt_len)
            prompt_lens = np.array(prompt_lens)
            max_prompt_len = np.max(prompt_lens)
            # sample diverse minority samples
            # each prompt starts with "target_variable is minority_class, a_feature is a_value"
            n_feature = len(feature_names)
            n_each_feature = int(np.ceil(gen_sample / n_feature))
            dfs = []
            for feature in feature_names:
                fea_dist = None
                if feature in numerical_names:
                    # the distribution of continuous variable is increased
                    fea_dist = X_y_imb_df[feature].to_list() + X_y_sm_df[feature].to_list()
                elif feature in categorical_names:
                    fea_dist = X_y_imb_df[feature].value_counts(1).to_dict()
                df_gen = imbllm.sample_imb(n_samples=n_each_feature, minority_cond=tokenized_minority,
                                           minority_class=minor_class, max_length=max_prompt_len,
                                           start_col=feature, start_col_dist=fea_dist)
                dfs.append(df_gen)
            X_y_minor_new = pd.concat(dfs)
            X_y_minor_new = X_y_minor_new.reset_index(drop=True)
            X_minor_new = X_y_minor_new.iloc[:, :-1]
            y_minor_new = X_y_minor_new.iloc[:, -1]
            X_bal_df = pd.concat([X_imb_enc, X_minor_new], axis=0, ignore_index=True)
            y_bal_df = pd.concat([y_imb_enc, y_minor_new], axis=0, ignore_index=True)
            X_train, y_train = read_data.encode_data(X_bal_df, y_bal_df,
                                                     numerical_names, categorical_names, encoding=c_encoding)
            class_dist = np.bincount(y_train)
            print("Imb_LLM class_dist: {}".format(class_dist))

        # encode balanced test set
        X_test, y_test = read_data.encode_data(X_test, y_test,
                                               numerical_names, categorical_names, encoding=c_encoding)

        # train a classifier
        # convert X_train and X_test to categorical data
        X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
        X_train.columns, X_test.columns = feature_names, feature_names
        y_train, y_test = pd.Series(y_train), pd.Series(y_test)
        for feature in categorical_names:
            X_train[feature] = X_train[feature].astype("category")
            X_test[feature] = X_test[feature].astype("category")
        for feature in numerical_names:
            X_train[feature] = X_train[feature].astype("float")
            X_test[feature] = X_test[feature].astype("float")
        xgb = XGBClassifier(random_state=run, enable_categorical=True, device="cuda")
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        acc_xgb = round(accuracy_score(y_test, y_pred), 4)
        if n_class == 2:
            f1_xgb = round(f1_score(y_test, y_pred), 4)
        elif n_class > 2:
            f1_xgb = round(f1_score(y_test, y_pred, average="weighted"), 4)

        labels = np.unique(y_train)
        # binarize y_test with shape (n_samples, n_classes)
        y_test = label_binarize(y_test, classes=labels)
        # binarize y_pred with shape (n_samples, n_classes)
        y_pred = label_binarize(y_pred, classes=labels)
        auc_xgb = round(roc_auc_score(y_test, y_pred, average="macro", multi_class="ovr"), 4)

        print("xgb run: {}, acc: {}, f1: {}, auc: {}".format(run, acc_xgb, f1_xgb, auc_xgb))
        acc_xgb_run[run] = acc_xgb
        f1_xgb_run[run] = f1_xgb
        auc_xgb_run[run] = auc_xgb

        # save result to text file
        if imbalance_method == "imb_llm":
            file_name = ("ds{}_tr{}_te{}_g_encode_{}_c_encode_{}_imb_{}_ra{}_smra{}_r{}_lo_{}".
                         format(dataset, train_size, test_size, g_encoding, c_encoding,
                                imbalance_method, imbalance_ratio, smote_ratio, n_run, location))
        else:
            file_name = ("ds{}_tr{}_te{}_g_encode_{}_c_encode_{}_imb_{}_ra{}_r{}_lo_{}".
                         format(dataset, train_size, test_size, g_encoding, c_encoding,
                                imbalance_method, imbalance_ratio, n_run, location))
        if run == (n_run - 1):
            # save result to text file
            with open('./results/accuracy_{}.txt'.format(file_name), 'w') as f:
                acc_xgb_avg, acc_xgb_std = round(np.mean(acc_xgb_run), 4), round(np.std(acc_xgb_run), 4)
                f1_xgb_avg, f1_xgb_std = round(np.mean(f1_xgb_run), 4), round(np.std(f1_xgb_run), 4)
                auc_xgb_avg, auc_xgb_std = round(np.mean(auc_xgb_run), 4), round(np.std(auc_xgb_run), 4)
                f.write("xgb - acc: {}, f1: {}, auc: {}\n".format(acc_xgb_run, f1_xgb_run, auc_xgb_run))
                f.write("xgb - acc_avg: {} ({}), f1_avg: {} ({}), auc_avg: {} ({})\n".
                        format(acc_xgb_avg, acc_xgb_std, f1_xgb_avg, f1_xgb_std, auc_xgb_avg, auc_xgb_std))

