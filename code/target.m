
function y=target(W1,W2,X,Y)
  pkg load geometry    
  # usage target(W1,W2,X,Y)
  # 
  # This function evaluates the sum of squares error for the
  # training set X,Y given the weight matrices W1 and W2 for 
  # a two-layered artificial neural network.
  # 
  # W1: weights matrix between input and hidden layer
  # W2: weights matrix between the hidden and the output layer
  # X:  training set holding on the rows the input data, plus a final column 
  #     equal to 1
  # Y:  labels of the training set
    
  dif=Y-predict(W1,W2,X);#difference between real and estimated value
  norma=(vectorNorm(dif)).^2; #norm² of each row 
  y=0.5*sum(norma); #error funtion value
endfunction;
#target(W1,W2,X,Y)
#waitforbuttonpress ();