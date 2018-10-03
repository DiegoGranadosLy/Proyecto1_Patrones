#[X,Y]=create_data(10000,3,'pie'); #obtaining trainnig data
#figure(1);
#plot_data(X,Y); #plotting de training data
#X=[ones(rows(X),1),X];  #bias
#W1=[1,1,1;1,1,3;1,2,23]; #example values
#W2=[1,2,2,3;1,6,3,1;1,2,3,5]; #example values


function [W1,W2] = gradient_descent(W1,W2,X,Y,batchSize=10000,lambda=0.001,maxIterations=2000,maxError=0.001)
    disp("entrenando la red neuronal...");
    disp(" ")
    disp("Al finalizar se mostrará un gráfico con la evolución del error ")
    disp(" ")
    ite=0; #number of iterations
    Error=100; #initial error
    newError=50;
    errorVec=[]; #vector for storing all the error values, one per iteration
    while(abs(Error-newError)>=maxError) #stop condition: when the difference betwen two error values is too small
        [nX,nY]=batch(X,Y,batchSize); #generates the data for each mini-batch iteration 
        Error=target(W1,W2,nX,nY); #initial value of the loss function
        [gW1,gW2]=gradtarget(W1,W2,nX,nY); #gradient for w1 y w2    
        W1=W1-lambda*gW1; #new W1 values using the gradient
        W2=W2-lambda*gW2; #new W2 values using the gradient
        newError=target(W1,W2,nX,nY); # current loss function value during training
        ite++;#current iteration during training
        errorVec=[errorVec,newError];
        if(ite>=maxIterations) #breaks the cycle if there are too much iterations
            break;
        endif
    endwhile
    figure(2);
    plot(1:ite,errorVec); #error vs iterations during the gradient descent
end






