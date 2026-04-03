"""
An example usage of Torchly where a neural network learns the function f(x,y) = xy - (x+y)
x ranges from  +5 to -5
y ranges from  +5 to -5
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchly import Model
import random

# Generate data
x = [random.randint(-5,5) for _ in range(1000)] 
y = [random.randint(-5,5) for _ in range(1000)] 

input_params = []
z = []

for i, j in zip(x, y): 
    fx = (i*j) - (i+j)
    input_params.append([i, j])  
    z.append([fx])               


model = Model([2, 64, 32, 1], activation="tanh")


model.train([input_params], z, epochs=500)


test = [[3, 3]]


pred = model.predict([test])

print("Prediction:", pred)
print("Actual:", (3*3) - (3+3))

# Extract weights

# Save & load
model.save("my_model.pt")



