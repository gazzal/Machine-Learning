import os

def get_instance_list(mode='train'):
    # Read files with list of train/test-instances
    with open(mode + '_instances.txt', 'r') as fh:
        list_insts = [line.strip() for line in fh.readlines()]
    return list_insts

def load_instances(instance_directory):
    # Here you have to insert code to load the data into the variables
    # For example:
    list_train_insts = get_instances('train')
    list_test_insts = get_instances('test')

    # Read file with labels
    with open(label_file, 'r') as fh:
        labels_dict = {k : int(v.strip('\n')) for k, v in [line.split('=') for line in fh.readlines()]}

    # Create X_train/X_test
    X_train, X_test, y_train, y_test = [], [], [], []
    for instance in os.listdir(instance_directory):
        with open(instance, 'r') as fh:
            content = fh.read()
        if instance in list_train_insts:
            X_train.append(content)
            y_train.append(labels_dict[instance])
        elif instance in list_test_insts:
            X_test.append(content)
            y_test.append(labels_dict[instance])
        else:
            print("{} not in any set (train or test)".format(instance))

    return X_train, y_train, X_test, y_test

def main():
    # First we create the training data
    X_train, y_train, X_test, y_test = load_instances()

    # Now, preprocessing (=feature extraction)
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Training the model
    model =
    model.fit(X_train, y_train)

    # Test the model
    predictions = model.predict(X_test)
    acc = len(y_test) / sum([p == r for p, r in zip(predictions, y_test)])
    print("The accuracy of the model is {}.".format(acc))

if __name__ == '__main__':
    main()
