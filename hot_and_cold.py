def neural_network(input,weight):
    prediction = input*weight
    return prediction



weight = 0.5
goal = 0.8
input = 0.5

learning_rate = 0.001

for iteration in range(1101):
    pred = input * weight
    error = (pred-goal)**2

    print("Error"+ str(error)+" Prediction "+str(pred))

    up_prediction = input*(weight+learning_rate)
    up_error = (up_prediction-goal) ** 2

    down_prediction = input*(weight-learning_rate)
    down_error = (down_prediction - goal) ** 2
    
    if(down_error<up_error):
        weight = weight - learning_rate
    if(up_error<down_error):
        weight = weight + learning_rate

