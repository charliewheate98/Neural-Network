# Neural-Network
A Neural Network that learns to output the correct value based on the inputs. E.g. if the input neurons are 0, 0, then the output should be 0, if 1, 0 then the output should be 1, if 1, 1 then the output should be 1.  The weights are generated in arbitrary fashion. I still plan on adding more to this project.
-------------------------------------------------------------------------------------------------------------------------------------------

A Neural Network you can say is a loose representation of the human brain that learns over a certain number of training passes. 
Depending on you type of network of course training will differ.
In my case I'm relatively new to neural networks so I started with a simple application that learns to output the mode of 2 values
with some rules however. 

if the values are:
  0, 0 = 0(Desired Output)
  1, 0 = 1(Desired Output)
  1, 1 = 1(Desired Output)
  
So this application trains the systems to print the correct value over 2000 training passes, by summing by the input and weights values
and putting the output value from this through a activation function which gives us the output/prediction. This activation function involves a mathmatical curve function to get a value between the range of -1 - 1, as I used the hyperbolic tangent function for this project due to the wider range and more arbitary outputs, therefore longer learning process. A Sigmoid function can of course be used and outputs values in the value of 0 - 1, therefore in theory the learning process would be quicker. 

Also the overall gradient is stronger for tanh, then sigmoid (derivates(rate of change) are steeper). Therefore a more accurate target near the end of the training passes.

Now its more than likely this output will be correct, so we do a process known as gradient descent and get the net error which is the 
gradient/range of the target output and the output the network gave us. I then back propagated to the hidden/middle layer and performed
the operations again including this gradient value in the calculation giving me a new value closer to what I expect due to the net error value.

I could talk for ages about Neural Networks, but till the next update feel free to download the code play around with it and have fun.

