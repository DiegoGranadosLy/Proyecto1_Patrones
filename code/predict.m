#[X,Y]=create_data(10); #obtainnig trainnig data
#plot_data(X,Y); #plotting de tarinig data
#W1=[1,1,1;1,1,1]; #example values
#W2=[1,2,2;1,3,3]; #example values

function g = sigmoid(X)
  g = 1 ./ (1 + e.^-X);
end

function y=predict(W1,W2,X)
    
  # usage predict(W1,W2,X)
  # 
  # This function propagates the input X on the neural network to
  # predict the output vector y, given the weight matrices W1 and W2 for 
  # a two-layered artificial neural network.
  # 
  # W1: weights matrix between input and hidden layer
  # W2: weights matrix between the hidden and the output layer
  # X:  Input vector, extended at its end with a 1
   
  X=[ones(rows(X),1),X];  #bias
  capa1=X*W1'; #first neuron layer input
  y1=sigmoid(capa1); #activation funtion
  y1=[ones(rows(y1),1),y1]; #bias
  capa2=y1*W2'; #second neuron layer input
  y=sigmoid(capa2); #activation funtion

  waitforbuttonpress ();
endfunction;
#predict(W1,W2,X)
