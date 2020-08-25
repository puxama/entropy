import entropy as en
import pandas as pd


if __name__ == '__main__':
    path = 'https://gist.githubusercontent.com/michhar/2dfd2de0d4f8727f873422c5d959fff5/raw/' \
            'fa71405126017e6a37bea592440b4bee94bf7b9e/titanic.csv'
    df_train = pd.read_csv(path)
    print("The entropy is: ", en.get_entropy(df_train['Survived'], 2))

    print("The information gain by variable is:")
    ig_list = en.get_information_gain(df_train[['Sex', 'Embarked', 'Pclass', 'Survived']], 'Survived', 2)
    for var in ig_list:
        print(var)

    en.plot_information_gain(df_train[['Sex', 'Embarked', 'Pclass', 'Survived']], 'Survived', 2)
    print("Categorical Entropy")
    en.plot_categorical_entropy(df_train['Embarked'], df_train['Survived'], 2)

    df_dataset = pd.concat([df_train['Embarked'], df_train['Survived']], axis=1)

    print(df_dataset.head())