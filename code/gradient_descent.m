[X,Y]=create_data(10000,3,'vertical'); #obtaining trainnig data
X=[ones(rows(X),1),X];  #bias
#plot_data(X,Y); #plotting de tarinig data
W1=[1,1,1;1,1,3;1,2,23]; #example values
W2=[1,2,2,3;1,6,3,1;1,2,3,5]; #example values


function [W1,W2] = gradient_descent(W1,W2,X,Y,batchSize=100000,lambda=0.002,maxIterations=2000,maxError=0.001)
    ite=0;
    Error=100;
    newError=50;
    errorVec=[];
    while(abs(Error-newError)>=maxError)
        [nX,nY]=batch(X,Y,batchSize);
        Error=target(W1,W2,nX,nY);
        [gW1,gW2]=gradtarget(W1,W2,nX,nY); #gradient for w1 y w2    
        W1=W1-lambda*gW1;
        W2=W2-lambda*gW2;
        newError=target(W1,W2,nX,nY);
        ite++;
        errorVec=[errorVec,newError];
        if(ite>=maxIterations)
            break;
        endif
    endwhile
    plot(1:ite,errorVec); #error vs iterations during the gradient descent
    waitforbuttonpress ();
end

[W1,W2]=gradient_descent(W1,W2,X,Y);
k=3;
x=linspace(-1,1,256);
[GX,GY]=meshgrid(x,x);
FX = [ones(size(GX(:)),1) GX(:) GY(:)];
FZ= predict(W1,W2,FX);
FZ=(FZ./sum(FZ,2))'
[maxprob,maxk]=max(FZ);
figure(k+2);

winner=flip(uint8(reshape(maxk,size(GX))),1);
cmap = [0,0,0; 1,0,0; 0,1,0; 0,0,1; 0.5,0,0.5; 0,0.5,0.5; 0.5,0.5,0.0];
wimg=ind2rgb(winner,cmap);
imshow(wimg);
title("Winner classes");
figure(k+3);

ccmap = cmap(2:1+k,:)
cwimg = ccmap'*FZ;
redChnl   = reshape(cwimg(1,:),size(GX));
greenChnl = reshape(cwimg(2,:),size(GX));
blueChnl  = reshape(cwimg(3,:),size(GX));

mixed = flip(cat(3,redChnl,greenChnl,blueChnl),1);
imshow(mixed);
title("Softmax classes");
waitforbuttonpress();

function w = packweights(W1,W2)
 w=[W1(:);W2(:)]
end

function [W1,W2] = unpackweights(w,rowsW1,colW1,rowsW2,colW2)
 W1=reshape(w(1:rowsW1*colW1),rowsW1,colW1)
 W2=reshape(w(rowsW1*colW1+1:end),rowsW2,colW2)
end