import random
import torch

def f(x):
    return (x+5)**2

# Derivative of f(x)
def dfdx(x):
    return 2*(x+5)

def minimize(f,dfdx,x0=0)->float:
    max_iter = 100
    x_guess = x0
    step_size = 0.1
    for _ in range(max_iter):
        grad = dfdx(x_guess)
        x_guess+= -step_size if grad > 0 else step_size
    return x_guess

def minimize_using_pytorch(f,dfdx,x0=0)->float:
    max_iter = 100
    x_guess = torch.tensor(x0, requires_grad=True)
    for _ in range(max_iter):
        loss = f(x_guess)
        loss.backward()
        step_size = 0.1 * abs(x_guess.grad)
        print(f"guess: {x_guess}, loss: {loss}, grad: {x_guess.grad}")
        if abs(x_guess.grad) < 0.01:
            break
        with torch.no_grad():
            x_guess += -step_size if x_guess.grad > 0 else step_size
            x_guess.grad.zero_()
        exit()



if __name__ == '__main__':
    #for x in range(-10, 10):
        #print(x, f(x),dfdx(x))
    x0= random.random()
    guess = minimize_using_pytorch(f,dfdx,x0)
    print(f"Minimum value of f(x) is {f(guess)} at x = {guess}")

    

    
    