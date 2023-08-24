# IMSL Lab - University of Notre Dame | University of St Andrews
# Author: Clemens JS Schaefer | Martin Schiemer

import numpy as np
import pandas as pd
from scipy import io
import random
from utils import utils


def get_hapt(remove_bad=True):
  class_name_translator = {1: "walking", 2: "walking upstairs",
                           3: "walking downstairs", 4: "sitting",
                           5: "standing", 6: "laying", 7: "stand to sit",
                           8: "sit to stand", 9: "sit to lie",
                           10: "lie to sit", 11: "stand to lie",
                           12: "lie to stand"}
  df = pd.read_csv("../Data/hapt_data/hapt_combined.csv")
  label_pos = "AID"
  person_column = "USER"
  if remove_bad:
      # remove class 8 because multiple users dont have it
      # remove users 7 and 28 beause they have missing classes
      bad_users = [7,28]
      for u in bad_users:
          df = df[df[person_column] != u]
      bad_classes = [8]
      for c in bad_classes:
          df = df[df[label_pos] != c]
  return df, person_column, label_pos, class_name_translator

def get_pamap(remove_bad=True):
  class_name_translator = {1: "lying", 2: "sitting", 3: "standing",
                           4: "walking", 5: "running", 6: "cycling",
                           7: "Nordic walking", 9: "watching TV",
                           10: "computer work", 11: "car driving",
                           12: "ascending stairs", 13: "descending stairs",
                           16: "vacuum cleaning", 17: "ironing", 
                           18: "folding laundry", 19: "house cleaning",
                           20: "playing soccer", 24: "rope jumping",
                           0: "other"}
  df = pd.DataFrame(io.loadmat('../Data/pamap/generated/pamap.mat')["data_pamap"])
  # df = pd.read_csv("../Data/pamap/generated/pamap.csv", header=None)
  label_pos = 243
  person_column = 244
  if remove_bad:
      # class 24 is not available for all users
      # user 3,4,9 have only limited classes
      bad_users = [3,4,9]
      for u in bad_users:
          df = df[df[person_column] != u]
      bad_classes = [24]
      for c in bad_classes:
          df = df[df[label_pos] != c]
  return df, person_column, label_pos, class_name_translator

def get_dsads():
  class_name_translator = {1: "sitting", 2: "standing", 3: "lying (back)",
                           4: "lying (right side)", 5: "ascending stairs",
                           6: "descending stairs", 7: "standing in elevator",
                           8: "moving in elevator",
                           9: "walking (parking lot)",
                           10: "walking on tredmill (flat)",
                           11: "walking on tredmill (incline)",
                           12: "running on tredmill",
                           13: "exercise on stepper",
                           14: "exercise (cross trainer)",
                           15: "cycling on bike (horizontal)",
                           16: "cycling on bike (vertical)", 17: "rowing",
                           18: "jumping", 19: "playing basketball"}
  # person column 407
  df = pd.read_csv("../Data/DSADS/generated/dsads.csv", header=None)
  # From data descriptions: Column 405 is the activity sequence indicating
  # the executing of activities (usually not used in experiments).
  
  df = utils.delete_pd_column(df, 405)
  df.rename(columns = {406:405}, inplace = True)
  df.rename(columns = {407:406}, inplace = True)
  label_pos = 405
  person_column = 406
  return df, person_column, label_pos, class_name_translator

def get_wisdm():
  class_name_translator = {1: "walking", 2: "jogging", 3: "stairs",
                           4: "sitting", 5: "standing", 6: "typing",
                           7: "teeth", 8: "soup", 9: "chips", 10: "pasta",
                           11: "drinking", 12: "sandwich", 13: "kicking",
                           14: "catch", 15: "dribbling", 16: "writing",
                           17: "clapping", 18: "folding"}
  headers = ["LBL", "X0", "X1", "X2", "X3", "X4", "X5", "X6",
              "X7", "X8", "X9", "Y0", "Y1", "Y2", "Y3", "Y4", "Y5",
              "Y6", "Y7", "Y8", "Y9", "Z0", "Z1", "Z2", "Z3", "Z4",
              "Z5", "Z6", "Z7", "Z8", "Z9", "XAVG", "YAVG", "ZAVG",
              "XPEAK", "YPEAK", "ZPEAK", "XABSOLDEV", "YABSOLDEV",
              "ZABSOLDEV", "XSTANDDEV", "YSTANDDEV", "ZSTANDDEV", "XVAR",
              "YVAR", "ZVAR", "XMFCC0", "XMFCC1", "XMFCC2", "XMFCC3",
              "XMFCC4", "XMFCC5", "XMFCC6", "XMFCC7", "XMFCC8", "XMFCC9",
              "XMFCC10", "XMFCC11", "XMFCC12", "YMFCC0", "YMFCC1",
              "YMFCC2", "YMFCC3", "YMFCC4", "YMFCC5", "YMFCC6", "YMFCC7",
              "YMFCC8", "YMFCC9", "YMFCC10", "YMFCC11", "YMFCC12",
              "ZMFCC0", "ZMFCC1", "ZMFCC2", "ZMFCC3", "ZMFCC4", "ZMFCC5",
              "ZMFCC6", "ZMFCC7", "ZMFCC8", "ZMFCC9", "ZMFCC10",
              "ZMFCC11", "ZMFCC12", "XYCOS", "XZCOS", "YZCOS", "XYCOR",
              "XZCOR", "YZCOR", "RESULTANT", "class"]
  label_pos = "LBL"
  df = pd.read_csv("../Data/wisdm/accel_wisdm.csv")
  df.columns = headers
  df["class"] = df["class"].apply(utils.delete_string_pos, args=(2,6))
  df["LBL"] = df["LBL"].apply(utils.delete_string_pos, args=(2,3))
  label_letters = "".join(list(sorted(df["LBL"].unique())))
  df["LBL"] = df["LBL"].apply(utils.change_letter_to_number,
                              alphabet=label_letters)
  data_types_dict = {'class': int}
  df = df.astype(data_types_dict)
  person_column = "class"
  
  return df, person_column, label_pos, class_name_translator

def get_data(d_name, TEST_SIZE, delete_class_column=False, user_test_set_size=0):
  if d_name.lower() == "wisdm":
      df, person_column, label_pos, class_name_translator = get_wisdm()
  elif d_name.lower() == "dsads":
      df, person_column, label_pos, class_name_translator = get_dsads()
  elif d_name.lower() == "pamap":
      df, person_column, label_pos, class_name_translator = get_pamap()
  elif d_name.lower() == "hapt":
      df, person_column, label_pos, class_name_translator = get_hapt()
  else:
      raise ValueError(f"Dataset {d_name} not found.")
    
  
  df[person_column] = df[person_column].astype("int")
  df[label_pos] = df[label_pos].astype("int")

  persons = df[person_column].unique()
  classes = df[label_pos].unique()
  if user_test_set_size:
      # separate second final test set that contains data from each user for each class
      user_test_indices = []
      for cl in classes:
          for person in persons:
              user_cl_indices = list(df[(df[person_column] == person) & (df[label_pos] == cl)].index)
              try:
                  test_smpls = random.sample(user_cl_indices,
                                             max(int(len(user_cl_indices)*user_test_set_size),1))
              except:
                  test_smpls = random.sample(user_cl_indices,
                                             max(int(len(user_cl_indices)*user_test_set_size),0))
              user_test_indices += test_smpls
      user_test_indices = list(set(user_test_indices))
      user_test_df = df.loc[user_test_indices]
      df.drop(user_test_indices, inplace=True,)
      df.reset_index(inplace=True, drop=True)
      user_test_df.reset_index(inplace=True, drop=True)
  else:
      user_test_df = None

  train_persons, test_persons = utils.random_split_list(list(persons), TEST_SIZE)
  print(f"Persons: {persons}, train: {sorted(train_persons)}, test: {sorted(test_persons)}")
  train_df = df.loc[df[person_column].isin(train_persons)]
  test_df = df.loc[df[person_column].isin(test_persons)]
  train_df.rename(columns = {person_column:'USER'}, inplace = True)
  test_df.rename(columns = {person_column:'USER'}, inplace = True)
  train_df.rename(columns = {label_pos:'LBL'}, inplace = True)
  test_df.rename(columns = {label_pos:'LBL'}, inplace = True)
  if user_test_set_size:
      user_test_df.rename(columns = {person_column:'USER'}, inplace = True)
      user_test_df.rename(columns = {label_pos:'LBL'}, inplace = True)
  if delete_class_column:
      train_df = utils.delete_pd_column(train_df, "USER")
      test_df = utils.delete_pd_column(test_df, "USER")
      if user_test_set_size:
          user_test_df = utils.delete_pd_column(user_test_df, "USER")
  print(train_df.head(2))
  label_pos = "LBL"
  return train_df, test_df, user_test_df, train_persons, test_persons, label_pos, class_name_translator


def transform_har_df_to_continual_dataset(df, test_df, label_col, user_col, args,
                                          user_test_df=None, class_name_translater=None,
                                          make_val=True, transform=None):
  if not args.user_discrimination:
      all_data_df = df.append(test_df)
      df, test_df = train_test_split(all_data_df, test_size=0.2, random_state=42)
  if make_val:
      val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
  else:
      val_df = None

  # randomly select parts of the data to be used for training
  if args.nr_classes_to_learn != "all":
      df, test_df, val_df, user_test_df = reduce_df_nr_column_values(df, test_df,
                                                                     label_col,
                                                                     args.nr_classes_to_learn,
                                                                     val_df=val_df,
                                                                     add_df=user_test_df,
                                                                     apply_to_test=True)
      
  if args.nr_users_to_learn != "all":
      df, test_df, val_df, user_test_df = reduce_df_nr_column_values(df, test_df,
                                                                     user_col,
                                                                     args.nr_users_to_learn,
                                                                     val_df=val_df,
                                                                     add_df=user_test_df,
                                                                     apply_to_test=True if not args.user_discrimination else False)
        
  features, labels, user_data = get_features_labels_users_from_df(df,
                                                                  label_col,
                                                                  user_col)
  if make_val:
      val_features, val_labels, val_user_data = get_features_labels_users_from_df(val_df,
                                                                                  label_col,
                                                                                  user_col)
  else:
      val_features, val_labels, val_user_data = None, None, None
  test_features, test_labels, test_user_data = get_features_labels_users_from_df(test_df,
                                                                                 label_col,
                                                                                 user_col)
  # test users are irrelevant

  if args.training_type in ["class", "task", "domain"]:
      dataset = ContinualLearningDataset(features, labels, test_features, test_labels,
                                         val_features, val_labels,
                                         users=user_data, transforms=transform,
                                         training_type=args.training_type,
                                         nr_classes_to_learn = args.nr_classes_to_learn,
                                         nr_users_to_learn=args.nr_users_to_learn,
                                         nr_classes_per_task=args.nr_classes_per_task,
                                         class_name_translater=class_name_translater,)
  elif args.training_type == "online":
      dataset = online_dataset.OnlineHARDataset(features, labels, user_data,
                                                test_features, test_labels,
                                                args.nr_classes_to_learn,
                                                args.nr_users_to_learn,
                                                args.perc_known_users,
                                                args.perc_known_classes,
                                                class_name_translater=class_name_translater,
                                                transform=transform)
  return dataset, user_test_df


def start_labels_at_zero(labels):
  labels = np.array(labels)
  labels -= np.min(labels)
  return labels

def make_lbls_continous(labels):
    labels = np.array(labels)
    nr_lbls = len(np.unique(labels))
    lbls = np.arange(nr_lbls)
    lbl_dict = dict(zip(np.unique(labels), lbls))
    labels = np.array([lbl_dict[lbl] for lbl in labels])
    return labels

def get_features_labels_users_from_df(df, label_col, user_col):
  label_data = df[label_col].values
  label_data =make_lbls_continous(label_data)
  label_data = start_labels_at_zero(label_data)
  user_data = df[user_col].values
  features = df.drop(columns=[label_col, user_col]).values.astype(np.float32)
  return features, label_data, user_data
