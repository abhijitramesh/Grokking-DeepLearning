number_of_toes = [0.8]
winning_toes = [1]
weight = 0.5
input = number_of_toes[0]
goal = winning_toes[0]

pred = input*weight
error = (pred-goal)**2

print(error)


