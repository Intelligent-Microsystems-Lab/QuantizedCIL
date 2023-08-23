import random

def random_split_list(a_list, split_perc):
    lista = random.sample(a_list, int(math.ceil(len(a_list) * (1 - split_perc))))
    listb = [item for item in a_list if item not in lista]
    return lista, listb


def delete_pd_column(dataframe, column):
    dataframe = dataframe.drop([column], axis=1)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def delete_string_pos(string, start, end=False):
    if end:
        return string[start:end]
    else:
        return string[start:]


def change_letter_to_number(letter, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                            start_with=1):
    return alphabet.index(letter)+start_with


def change_number_to_letter(number, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return alphabet[number-1]