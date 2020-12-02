input = 2
goal = 0.8
weight = 0.5
alpha = 0.1

for iteration in range(20):
    pred = input*weight
    error = (pred-goal)**2
    divergence = input*(pred-goal)
    weight = weight - (alpha*divergence)

    print("Error"+str(error)+" Prediction "+str(pred))

