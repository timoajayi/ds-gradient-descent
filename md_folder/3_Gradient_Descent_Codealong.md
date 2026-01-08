# Write your own Gradient Descent!

## The Algorithm

A simple gradient Descent Algorithm looks like this:


1. Obtain a function to minimize F(x)

2. Initialize a value x from which to start the descent or optimization from

3. Specify a learning rate that will determine how much of a step to descend by or how quickly you converge to the minimum value

4. Obtain the derivative of that value x (the descent)

5. Proceed to descend by the derivative of that value multiplied by the learning rate

6. Adjust the value of x

7. Check your stop condition to see whether to stop

8. If condition satisfied, stop. If not, proceed to step 4 with the new x value and keep repeating algorithm

## Let's implement this in Python

We will implement a simple representation of gradient descent using python. 

We will create an arbitrary loss function and attempt to find a local minimum value the range from -1 and 3 for that function 
f(x) = x³ — 3x² + 7 .

### Step 1

We will first visualize this function with a set of values ranging from -1 and 3 


```python
# Your code here
# creating the function and plotting it 

def function(x):
    pass


# Get 1000 evenly spaced numbers between -1 and 3 


# Plot the curve
```

### Step 2

We will then proceed to make two functions for the gradient descent implementation.

The first is a derivative function: 

This function takes in a value of x and returns its derivative based on the initial function we specified. It is shown below:


```python
def deriv(x):
    
    '''
    Description: This function takes in a value of x and returns its derivative based on the 
    initial function we specified.
    
    Arguments:
    
    x - a numerical value of x 
    
    Returns:
    
    x_deriv - a numerical value of the derivative of x
    
    '''
    
    
    return x_deriv
```

#### Step 3

The second is a Step function: 


This is the function where the actual gradient descent takes place. 



This function takes in an initial or previous value for x, updates it based on the step taken via the descent multiplied by the learning rate and outputs the most minimum value of x that reaches the stop condition. 



For our stop condition, we are going to use a precision stop.



This means that when the absolute difference between our old and updated x is smaller than a value, the algorithm should stop. 



The function will also print out the minimum value of x as well as the number of steps or descents it took to reach that value.


```python
# Your code here

def step(x_new, x_prev, precision, l_r):
    
    '''
    Description: This function takes in an initial or previous value for x, updates it based on 
    steps taken via the learning rate and outputs the most minimum value of x that reaches the precision satisfaction.
    
    Arguments:
    
    x_new - a starting value of x that will get updated based on the learning rate
    
    x_prev - the previous value of x that is getting updated to the new one
    
    precision - a precision that determines the stop of the stepwise descent 
    
    l_r - the learning rate (size of each descent step)
    
    Output:
    
    1. Prints out the latest new value of x which equates to the minimum we are looking for
    2. Prints out the the number of x values which equates to the number of gradient descent steps
    3. Plots a first graph of the function with the gradient descent path
    
    '''
    
    # create empty lists where the updated values of x and y will be appended during each iteration
    
    
    # keep looping until your desired precision
    
        
        # change the value of x
        
        
        # get the derivation of the old value of x
        
        
        # get your new value of x by substracting the multiplication of the derivative and the learning rate from the previous x 
        
        
        # append the new value of x to a list of all x for later visualization of path
        
        
        # append the new value of y to a list of all ys for later visualization of path
       

    print ("Local minimum occurs at: "+ str(x_new))
    print ("Number of steps: " + str(len(x_list)))
    
    # Create plot to show Gradient descent, 
    
    plt.show()
```

#### Step 4

Now we will use our two functions and see if they are working correctly.


```python
# Implement gradient descent (all the arguments are arbitrarily chosen)

step(0.5, 0, 0.001, 0.05)
```

    Local minimum occurs at: 0.5



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[4], line 3
          1 # Implement gradient descent (all the arguments are arbitrarily chosen)
    ----> 3 step(0.5, 0, 0.001, 0.05)


    Cell In[3], line 49, in step(x_new, x_prev, precision, l_r)
         27 # create empty lists where the updated values of x and y will be appended during each iteration
         28 
         29 
       (...)
         44     
         45     # append the new value of y to a list of all ys for later visualization of path
         48 print ("Local minimum occurs at: "+ str(x_new))
    ---> 49 print ("Number of steps: " + str(len(x_list)))
         51 # Create plot to show Gradient descent, 
         53 plt.show()


    NameError: name 'x_list' is not defined

