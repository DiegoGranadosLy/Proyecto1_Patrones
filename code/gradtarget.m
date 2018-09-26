
function [gW1,gW2]=gradtarget(W1,W2,X,Y)

  # usage gradtarget(W1,W2,X,Y)
  # 
  # This function evaluates the gradient of the target function on W1 and W2.
  # 
  # W1: weights matrix between input and hidden layer
  # W2: weights matrix between the hidden and the output layer
  # X:  training set holding on the rows the input data, plus a final column 
  #     equal to 1
  # Y:  labels of the training set
  
  capa1=X*W1'; #first neuron layer input
  y1=sigmoid(capa1); #activation funtion
  y1=[ones(rows(y1),1),y1]; #bias
  capa2=y1*W2'; #second neuron layer input
  y=sigmoid(capa2); #activation funtion

  delta2=-(Y-y).*(y.*(1-y));
  gW2=(y1'*delta2)';

  delta1= (delta2*W2).*(y1.*(1-y1));
  gW1=(X'*delta1)';
  gW1=gW1(2:rows(gW1),:);
endfunction;

#gradtarget(W1,W2,X,Y,10);
#waitforbuttonpress ();
#https://aimatters.wordpress.com/2016/01/03/a-simple-neural-network-in-octave-part-2/
#https://gigadom.in/2018/01/11/deep-learning-from-first-principles-in-python-r-and-octave-part-2/
#http://georgepavlides.info/matrix-based-implementation-neural-network-back-propagation-training-matlab-octave-approach/