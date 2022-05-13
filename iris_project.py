# Iris Classification

import math
import turtle

def create_table(filename):
    """
    sig: str -> tuple(list(float), list(float), list(str))
    Given a file name, reads the file into a tuple containing
    two lists of type float and one list of type string.
    The features of the dataset should be of type float
    and the label should be of type string. 
    """
    train_file = open(filename, "r")
    lst_lengths = []
    lst_widths = []
    lst_labels = []
    for line in train_file:
        ind_row = line.split(",")
        lst_lengths.append(float(ind_row[0]))
        lst_widths.append(float(ind_row[1]))
        lst_labels.append(ind_row[2].strip())
    train_tuple = (lst_lengths, lst_widths, lst_labels)
    train_file.close()
    return train_tuple

def print_range_max_min(data):
    """
    sig: tuple(list(float), list(float)) -> NoneType
    Prints the max, min and range of both features in the dataset.
    """
    max_length = max(data[0])
    min_length = min(data[0])
    range_length = max_length - min_length
    max_width = max(data[1])
    min_width = min(data[1])
    range_width = max_width - min_width
    print("Feature 1 - min:", str(min_length), "max:", str(max_length),\
          "range:", str(range_length))
    print("Feature 2 - min:", str(min_width), "max:", str(max_width),\
          "range:", str(range_width))
    return None

def find_mean(feature):
    """
    sig: list(float) -> float
    Returns the mean of the feature.
    """
    sum_nums = 0.0
    for num in feature:
        sum_nums += num
    mean = sum_nums/len(feature)
    return mean

def find_std_dev(feature, mean):
    """
    sig: list(float), float -> float
    Returns the standard deviation of the feature. 
    """
    sq_sum_diff = 0.0
    for num in feature:
        sq_sum_diff += (num - mean)**2
    std_dev = math.sqrt(sq_sum_diff/len(feature))
    return std_dev
    
def normalize_data(data):
    """
    sig: tuple(list(float), list(float), list(str)) -> NoneType
    Prints the mean and standard deviation for each feature.
    Then, normalizes the features in the dataset by
    rescaling all the values in a particular feature
    in terms of a mean of 0 and a standard deviation of 1.
    Afterwards, prints the mean and the standard deviation for each feature, now normalized,
    to make sure each of the features display a mean of 0
    or very close to 0 and a standard deviation of 1 or very close to 1. 
    """
    # Recall: feature 1 references the lengths.
    feat1_mean = find_mean(data[0])
    feat1_stddev = find_std_dev(data[0], feat1_mean)
    print("Feature 1 - mean:", str(feat1_mean), "standard deviation:", str(feat1_stddev))
    # Here, I am modifying a list of lengths by reassigning values at index "i" with their normalized values.
    for i in range(len(data[0])):
        data[0][i] = (data[0][i] - feat1_mean)/feat1_stddev 
    feat1_norm_mean = find_mean(data[0])
    feat1_norm_stddev = find_std_dev(data[0], feat1_norm_mean)
    print("Feature 1 after normalization - mean:", str(feat1_norm_mean), "standard deviation:", str(feat1_norm_stddev))
    # Recall: feature 2 references the widths.
    feat2_mean = find_mean(data[1])
    feat2_stddev = find_std_dev(data[1], feat2_mean)
    print("Feature 2 - mean:", str(feat2_mean), "standard deviation:", str(feat2_stddev))
    # Here, I am modifying a list of widths by reassigning values at index "i" with their normalized values.
    for i in range(len(data[1])): 
        data[1][i] = (data[1][i] - feat2_mean)/feat2_stddev
    feat2_norm_mean = find_mean(data[1])
    feat2_norm_stddev = find_std_dev(data[1], feat2_norm_mean)
    print("Feature 2 after normalization - mean:", str(feat2_norm_mean), "standard deviation:", str(feat2_norm_stddev))
    return None

def make_predictions(train_set, test_set):
    """
    sig: tuple(list(float), list(float), list(str)), tuple(list(float), list(float), list(str)) -> list(str)
    For each observation in the test set, check all of
    the observations in the training set to see which is the `nearest neighbor.'
    The function makes a call to the function find_dist.
    Accumulates a list of predicted iris types for each of the test set
    observations. Returns this prediction list.
    """
    pred_lst = []
    test_set_lengths = test_set[0]
    test_set_widths = test_set[1]
    test_set_labels = test_set[2]
    lst_tuples_test = []
    # In this for-loop, I'm creating a tuple(list(lengths), list(widths)) of values from the test_set.
    for i in range(len(test_set_widths)):
        lst_tuples_test.append((test_set_lengths[i], test_set_widths[i]))
    train_set_lengths = train_set[0]
    train_set_widths = train_set[1]
    train_set_labels = train_set[2]
    lst_tuples_train = []
    # In this for-loop, I'm creating a tuple(list(lengths), list(widths)) of values from the train_set.
    for i in range(len(train_set_widths)):
        lst_tuples_train.append((train_set_lengths[i], train_set_widths[i]))
    for (length1, width1) in lst_tuples_test:
        '''
        The outer part of this for-loop is preparing each (length, width) coordinate of the data_set needed to be tested.
        '''
        index = 0
        nearest_neighbor = ""
        smallest_distance = 100000.0
        x1_coord = length1
        y1_coord = width1
        for (length2, width2) in lst_tuples_train:
            '''
            The inner part of this for-loop is comparing the given coordinate from above (we need to label/predict yet) to each and every trainee coordinate-point.
            This is in effort to find the trainee coordinate point that has the smallest distance and is therefore the closest coordinate-point to the
            coordinate point of the "unknown" iris plant.  The "unknown" iris plant will then adopt it's nearest_neighbor's classification/label.
            '''
            x2_coord = length2
            y2_coord = width2
            distance = find_dist(x1_coord, y1_coord, x2_coord, y2_coord)
            if distance < smallest_distance:
                nearest_neighbor = train_set_labels[index]
                smallest_distance = distance
            index += 1
        pred_lst.append(nearest_neighbor)
    return pred_lst

def find_dist(x1, y1, x2, y2):
    """
    sig: float, float, float, float -> float
    Returns the Euclidean distance between two points (x1, y1), (x2, y2).
    """
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def find_error(test_data, pred_lst):
    """
    sig: tuple(list(float), list(float), list(str)), lst(str)) -> float
    Checks the prediction list against the actual labels for
    the test set to determine how many errors were made.
    Returns a percentage of how many observations in the
    test set were predicted incorrectly. 
    """
    true_labels_lst = test_data[2]
    '''
    Above, I separated the list of the correct labels from the test_data tuple
    so that I can more easily use this list to compare it's labels to my predicted values in the pred_lst -
    for style and in effort to simplify reading the code.
    '''
    index = 0
    num_wrong = 0
    for label in pred_lst:
        if label != true_labels_lst[index]:
            num_wrong += 1
        index += 1
    perc_error = (num_wrong/len(pred_lst)) * 100
    return perc_error

def plot_data(train_data, test_data, pred_lst):
    """
    sig: tuple(list(float), list(float), list(str)), tuple(list(float), list(float), list(str)), list(str)
        -> NoneType
    Plots the results using the turtle module. 
    Plots each observation from the training set on the plane, using a circle shape
    and a different color for each type of iris. Uses the value of the first feature
    for the x-coordinate and the value of the second feature for the y-coordinate.
    Recall that the features have been normalized to have a mean
    of 0 and a standard deviation of 1. Thus, we will need to `stretch' the features across
    the axes to make the best use of the 500 x 500 window.
    A square will indicate that the value is a prediction.
    Incorrect predictions that were made for the test set will be in red, also using a
    square to indicate that it was a prediction.
    """
    draw_grid()
    # Here, I am plotting the trainee coordinates.
    for i in range(len(train_data[0])):
        turtle.setposition(125 * train_data[0][i] + 250, 125 * train_data[1][i] + 250)
        '''
        Accounting for my origin actually being at (250, 250) instead of (0,0)
        and to guarantee my points fill the 500 x 500 unit window
        I am stretching/enlarging my values by "125" and shifting the points by 250.
        '''
        if train_data[2][i] == 'Iris-setosa':
            turtle.color("purple")
        elif train_data[2][i] == 'Iris-versicolor':
            turtle.color("blue")
        elif train_data[2][i] == 'Iris-virginica':
            turtle.color("green")
        turtle.pendown()
        turtle.dot(10)
        turtle.penup()
    for x in range(len(test_data[0])):
        turtle.setposition(125 * test_data[0][x] + 250, 125 * test_data[1][x] + 250)
        if test_data[2][x] == pred_lst[x]:
            if test_data[2][x] == 'Iris-setosa':
                turtle.color("purple")
            elif test_data[2][x] == 'Iris-versicolor':
                turtle.color("blue")
            elif test_data[2][x] == 'Iris-virginica':
                turtle.color("green")
        else:
            turtle.color("red")
        turtle.pendown()
        draw_square()
        turtle.penup()
    draw_key()
    return None
            
def draw_key():
    """
    sig: () -> NoneType
    Draws the legend for the plot indicating which group is shown by each color/shape combination.  
    """
    down_shift = 0
    # This for-loop is designated to drawing the shapes (with the correct color) for the key.
    for i in range(7):
        if i == 0 or i == 3:
            turtle.color("purple")
        elif i == 1 or i == 4:
            turtle.color("blue")
        elif i == 2 or i == 5:
            turtle.color("green")
        else:
            turtle.color("red")
        turtle.goto(50, 475 + down_shift)
        turtle.pendown()
        if i == 0 or i == 1 or i == 2:
            turtle.dot(10)
        else:
            draw_square()
        turtle.penup()
        down_shift -= 15
    turtle.color("black")
    down_shift = 0
    # This for-loop is designated to writing the names for the key.
    for i in range(7):
        turtle.goto(70, 465 + down_shift)
        if i == 0:
            turtle.write("Iris-setosa")
        elif i == 1:
            turtle.write("Iris-versicolor")
        elif i == 2:
            turtle.write("Iris-virginica")
        elif i == 3:
            turtle.write("Predicted Iris-setosa")
        elif i == 4:
            turtle.write("Predicted Iris-versicolor")
        elif i == 5:
            turtle.write("Predicted Iris-virginica")
        elif i == 6:
            turtle.write("Predicted Incorrectly")
        down_shift -= 15
    return None

def draw_grid():
    '''
    sig: () -> NoneType
    Sets the turtle window size to 500 x 500.
    Draws and labels the x and y axes in the window.
    '''
    turtle.setworldcoordinates(0, 0, 500, 500)
    turtle.penup()
    turtle.goto(-5,250)
    turtle.pendown()
    turtle.forward(500)
    turtle.penup()
    turtle.setposition(250,-5)
    turtle.left(90)
    turtle.pendown()
    turtle.forward(500)
    turtle.penup()
    turtle.setposition(455,255)
    turtle.write("petal length")
    turtle.penup()
    turtle.setposition(255,5)
    turtle.write("petal width")
    turtle.hideturtle()
    return None

def draw_square():
    '''
    sig: () -> NoneType
    A separate function designated solely to drawing the square data points.
    '''
    turtle.pendown()
    turtle.begin_fill()
    turtle.right(90)
    turtle.forward(5)
    turtle.right(90)
    turtle.forward(5)
    turtle.right(90)
    turtle.forward(5)
    turtle.right(90)
    turtle.forward(5)
    turtle.end_fill()
    return None

def main():
    """
    sig: () -> NoneType
    The main body of the program. It will use the other
    functions to load the data, process the training set,
    analyze the test set, and display its conclusions.
    """
    train_data = create_table("iris_train.csv")
    print_range_max_min(train_data[:2])
    print()
    normalize_data(train_data)
    test_data = create_table("iris_test.csv")
    print()
    normalize_data(test_data)
    pred_lst = make_predictions(train_data, test_data)
    error = find_error(test_data, pred_lst)
    print()
    print("The error percentage is: ", error)
    plot_data(train_data, test_data, pred_lst)
    
main()
