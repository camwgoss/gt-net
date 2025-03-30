from sklearn.model_selection import train_test_split


def split_data(data: list):
    '''
    Randomly split data into 70% training, 15% evaluation, and 15% test data. 
    This function is deterministic, so it can be called for separately for
    image data and mask data while maintaining perfect alignment between images
    and masks.
    Arguments:
        data: Data to split.
    Returns:
        train, evaluation, test: Lists containing train, evaluation, and test data.
    '''

    # specify random_state to make split deterministic
    train, other = train_test_split(data, train_size=0.7, random_state=333)

    # split other into 50%-50% evaluation and test, which is 15%-15% of data
    evaluation, test = train_test_split(other, test_size=0.5, random_state=333)

    return train, evaluation, test
