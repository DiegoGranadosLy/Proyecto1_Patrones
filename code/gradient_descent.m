#[X,Y]=create_data(5000,3,'pie'); #obtaining trainnig data
#figure(1);
#plot_data(X,Y); #plotting de training data
#X=[ones(rows(X),1),X];  #bias
#W1=[1,1,1;1,1,3;1,2,23]; #example values
#W2=[1,2,2,3;1,6,3,1;1,2,3,5]; #example values


function [W1,W2] = gradient_descent(W1,W2,X,Y,batchSize=100000,lambda=0.002,maxIterations=2000,maxError=0.001)
    disp("entrenando la red neuronal...");
    disp("Al finalizar se mostrará un gráfico con la evolución del error ")
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
    waitforbuttonpress ();
end

function v = visualization(W1,W2,X,Y)
    k=3; #number of classes 
    [W1,W2]=gradient_descent(W1,W2,X,Y,100000); #training the neural network
    #preparing de input data for a 256x256 image#
    x=linspace(-1,1,256);
    [GX,GY]=meshgrid(x,x);
    FX = [ones(size(GX(:)),1) GX(:) GY(:)];
    #############################################
    #generating the prediction for the input data#
    disp("realizando la clasificación...");
    FZ= predict(W1,W2,FX);
    FZ=(FZ./sum(FZ,2))'; #each input vector in divided by the sum of all its elements, so each vector sum is 1
    #image realated operations#
    cmap = [0,0,0; 1,0,0; 0,1,0; 0,0,1; 0.5,0,0.5; 0,0.5,0.5; 0.5,0.5,0.0];
    figure(3);
    ccmap = cmap(2:1+k,:);
    cwimg = ccmap'*FZ;
    redChnl   = reshape(cwimg(1,:),size(GX));
    greenChnl = reshape(cwimg(2,:),size(GX));
    blueChnl  = reshape(cwimg(3,:),size(GX));
    mixed = flip(cat(3,redChnl,greenChnl,blueChnl),1);
    imshow(mixed); #showing the result
    ######################################################
    title("Clasificación de cada punto en una imagen");
    waitforbuttonpress();

end

visualization(W1,W2,X,Y);

function w = packweights(W1,W2)
 w=[W1(:);W2(:)]
end

function [W1,W2] = unpackweights(w,rowsW1,colW1,rowsW2,colW2)
 W1=reshape(w(1:rowsW1*colW1),rowsW1,colW1)
 W2=reshape(w(rowsW1*colW1+1:end),rowsW2,colW2)
end

