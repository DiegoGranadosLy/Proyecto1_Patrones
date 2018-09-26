#[X,Y]=create_data(10000); #obtaining trainnig data
#X=[ones(rows(X),1),X];  #bias
#plot_data(X,Y); #plotting de tarinig data
#W1=[1,1,1;1,1,3;1,2,23]; #example values
#W2=[1,2,2,3;1,6,3,1;1,2,3,5]; #example values
function w = packweights(W1,W2)
 w=[W1(:);W2(:)]
end

function [W1,W2] = unpackweights(w,rowsW1,colW1,rowsW2,colW2)
 W1=reshape(w(1:rowsW1*colW1),rowsW1,colW1)
 W2=reshape(w(rowsW1*colW1+1:end),rowsW2,colW2)
end


function w = gradient_descent(W1,W2,X,Y,batchSize=1000,lambda=0.02,maxIterations=1000,maxError=0.0001)
    ite=0;
    Error=100;
    newError=50;
    errorVec=[];
    while(Error-newError>=maxError)
        [X,Y]=batch(X,Y,batchSize);
        Error=target(W1,W2,X,Y);
        [gW1,gW2]=gradtarget(W1,W2,X,Y); #gradient for w1 y w2    
        W1=W1-lambda*gW1;
        W2=W2-lambda*gW2;
        newError=target(W1,W2,X,Y);
        ite++;
        errorVec=[errorVec,newError];
        
    endwhile
    #plot(1:ite,errorVec); #error vs iterations during the gradient descent algorithm
end

#gradient_descent(W1,W2,X,Y);
#waitforbuttonpress ();