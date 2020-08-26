import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt


def get_entropy(population, base=math.e):
    """
    A function to calculate entropy of the population

    Parameters:
        population : list
            labels of the population

        base : float, optional
            base of logarithm, it is recommended to use the number
            of unique labels.

            default = math.e
    Returns: float
        Value of the entropy
    """

    size_population = len(population)
    labels, counts = np.unique(population, return_counts=True)

    if len(labels) <= 1:
        return 0

    entropy = 0
    for i in counts / size_population:
        entropy -= i * math.log(i, base)

    return round(entropy, 2)


def _auto_label(rects, ax, category_list):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect, category in zip(rects, category_list):
        height = rect.get_height()
        ax.annotate(category,
                    xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', size=20)


def _get_categorical_entropy(categorical_var, population, base=math.e):
    """
    A function to calculate entropy of the population by categorical var

    Parameters:

        categorical_var: list
            values of the categorical value
        population : list
            labels of the population

        base : float, optional
            base of logarithm, it is recommended to use the number
            of unique labels.

            default = math.e

    Returns: list of tuple (category, entropy, proportion)

    """
    if len(categorical_var) != len(population):
        raise ValueError("The size of the input lists is different")

    category_list, counts = np.unique(categorical_var, return_counts=True)

    results = []
    entropy_list = []
    for category in category_list:
        subset = \
            np.array(population)[np.where(np.array(categorical_var) == category)[0]]

        entropy_list.append(get_entropy(subset, base=base))

        results.append((category, entropy_list[-1],
                        round(len(subset) / len(population), 2)))

    return results


def get_categorical_entropy(df_dataset, population_name, base=math.e):
    """
    A function to calculate entropy of the population by categorical var.
    If categorical variable has missing values, there are removed

    Parameters:

        df_dataset: pandas.Dataframe
            DataFrame with 2 columns: categorical variable and Y
        population_name : name of column Y in DataFrame
            labels of the population

        base : float, optional
            base of logarithm, it is recommended to use the number
            of unique labels.

            default = math.e

    Returns: list of tuple (category, entropy, proportion)

    """
    try:
        categorical_name = [var for var in df_dataset if population_name != var][0]
    except ValueError:
        print("No found categorical variable")
        return None

    if df_dataset[population_name].isnull().sum() > 0:
        raise ValueError("Variable Y has missing values")

    df_dataset = df_dataset.loc[~df_dataset[categorical_name].isnull(), ]
    results = _get_categorical_entropy(df_dataset[categorical_name],
                                       df_dataset[population_name],
                                       base)

    return results


def _get_information_gain(categorical_var, population, base=math.e):
    """
    A function to get the information gain bases on a entropy measure

    Parameters:
        categorical_var: list
            values of the categorical value
        population : list
            labels of the population

        base : float, optional
            base of logarithm, it is recommended to use the number
            of unique labels.

            default = math.e
    Returns: float
            Value of information gain
    """

    if len(categorical_var) != len(population):
        raise ValueError("The size of the input lists is different")

    entropy_before = get_entropy(population, base)

    category_entropy_list = _get_categorical_entropy(categorical_var, population, base)

    entropy_after = np.sum([category_entropy[1] * category_entropy[2]
                            for category_entropy in category_entropy_list])

    return round(entropy_before - entropy_after, 2)


def get_information_gain(df_dataset, population_name, base=math.e):
    """
    A function to get the information gain bases on a entropy measure
    Parameters:
        df_dataset: pandas.Dataframe
            DataFrame with 2 columns: categorical variable and Y
        population_name : name of column Y in DataFrame
            labels of the population
        base : float, optional
            base of logarithm, it is recommended to use the number
            of unique labels.

            default = math.e

    Returns: float
            Value of information gain
    """

    category_names_list = [var for var in df_dataset if population_name != var]
    if len(category_names_list) == 0:
        raise ValueError("No categorical variable found")

    if df_dataset[population_name].isnull().sum() > 0:
        raise ValueError("Variable Y has missing values")

    information_gain_list = []
    for category_name in category_names_list:
        df_subset = df_dataset.loc[~df_dataset[category_name].isnull(), ]
        gain_information = _get_information_gain(df_subset[category_name],
                                                 df_subset[population_name],
                                                 base)
        information_gain_list.append((category_name, gain_information))

    return information_gain_list


def plot_categorical_entropy(categorical_var, population, base=math.e, name_categorical_var=''):
    """
    Plot a entropy with proportion of categorical var

    Parameters:

        categorical_var: list
            values of the categorical value
        population : list
            labels of the population

        base : float, optional
            base of logarithm, it is recommended to use the number
            of unique labels.

            default = math.e

        name_categorical_var: string
            name that describe the categorical var, it is used for the plot's title

    Returns: No return

    """
    df_dataset = pd.concat([categorical_var, population], axis=1)
    df_dataset = df_dataset.loc[~df_dataset.iloc[:, 0].isnull()]
    category_entropy_list = _get_categorical_entropy(df_dataset.iloc[:, 0],
                                                     df_dataset.iloc[:, 1],
                                                     base)

    plt.rcParams["figure.figsize"] = [10, 6]

    category_list = []
    entropy_list = []
    proportion_list = []
    for category_entropy in category_entropy_list:
        category_list.append(category_entropy[0])
        entropy_list.append(category_entropy[1])
        proportion_list.append(category_entropy[2])

    data = entropy_list
    widths = proportion_list
    left = [0]
    for var in widths[:-1]:
        left.append(round(var + left[-1], 3))

    fig, ax = plt.subplots()
    rects = ax.bar(left, data, width=widths,
                   alpha=0.6, align='edge', edgecolor='k', linewidth=2)

    _auto_label(rects, ax, category_list)

    # Add title and axis names
    plt.title('Entropy and prevalence of values: ' + name_categorical_var, fontsize=20)
    plt.xlabel('Proportion', fontsize=20)
    plt.ylabel('Entropy', fontsize=20)
    plt.show()


def plot_information_gain(df_dataset, population_name, base=math.e):
    """
    A function to get the information gain bases on a entropy measure
    Parameters:
        df_dataset: pandas.Dataframe
            DataFrame with 2 or more columns: categorical variable and Y
        population_name : name of column Y in DataFrame
            labels of the population
        base : float, optional
            base of logarithm, it is recommended to use the number
            of unique labels.

            default = math.e

    Returns: float
            Value of information gain
    """
    information_gain_list = get_information_gain(df_dataset, population_name, base)
    information_gain_list.sort(key=lambda x: x[1], reverse=True)

    people = list(zip(*information_gain_list))[0]
    score = list(zip(*information_gain_list))[1]
    x_pos = np.arange(len(people))

    plt.bar(x_pos, score, align='center')
    plt.title('Information Gain by Variable')
    plt.xticks(x_pos, people)
    plt.ylabel('Information Gain')
    plt.show()
