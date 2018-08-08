import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def get_instance_list(mode='train'):
    # Read files with list of train/test-instances
    with open(mode + '_instances.txt', 'r') as fh:
        list_insts = [line.strip() for line in fh.readlines()]
    return list_insts

def load_instances(instance_directory, label_file='instance_labels.txt'):
    # Here you have to insert code to load the data into the variables
    # For example:
    list_train_insts = get_instance_list('train')
    list_test_insts = get_instance_list('test')

    # Read file with labels
    with open(label_file, 'r') as fh:
        labels_dict = {k : int(v.strip('\n')) for k, v in [line.split('=') for line in fh.readlines()]}

    # Create X_train/X_test
    X_train, X_test, y_train, y_test = [], [], [], []
    for instance in os.listdir(instance_directory):
        with open(os.path.join(instance_directory, instance), 'r') as fh:
            content = fh.read()
        if instance in list_train_insts:
            X_train.append(content)
            y_train.append(labels_dict[instance])
        elif instance in list_test_insts:
            X_test.append(content)
            y_test.append(labels_dict[instance])
        else:
            print("{} not in any set (train or test)".format(instance))

    print("Number of train samples: {}, number of test samples: {}".format(len(y_train), len(y_test)))

    return X_train, y_train, X_test, y_test

def main():
    # First we create the training data
    X_train, y_train, X_test, y_test = load_instances("instances")
    #  print(X_train)

    # X are now the text-files as raw strings, preprocessing (=feature extraction)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    print("The shapes of the X feature-matrix are: train: {}, test: {}".format(X_train.shape, X_test.shape))

    # Training the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Test the model
    predictions = model.predict(X_test)
    correct = sum([p == r for p, r in zip(predictions, y_test)])
    acc = correct / len(y_test)
    # print(list(zip(predictions, y_test)))
    print("The accuracy of the model is {}/{}={}.".format(correct, len(y_test), acc))

if __name__ == '__main__':
    main()
