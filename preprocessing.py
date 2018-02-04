import os
import cPickle


def pickle_call(file_name):
    # print os.path.isfile(file_name)
    if os.path.isfile(file_name):
        with open(file_name, "rb") as input_file:
            return cPickle.load(input_file)
    else:
        return None


def pickle_dump(file_name, data):
    with open(file_name, "wb") as output_file:
        cPickle.dump(data, output_file)


