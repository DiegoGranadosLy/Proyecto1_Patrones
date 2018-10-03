pkg load optim;
pkg load geometry

[X,Y]=create_data(10000,3,'pie'); #obtaining trainnig data
#figure(1);
#plot_data(X,Y); #plotting de training data
X=[ones(rows(X),1),X];  #bias
W1=[1,1,1;1,1,3;1,2,23]; #example values
W2=[1,2,2,3;1,6,3,1;1,2,3,5]; #example values
wSize=[rows(W1),columns(W1),rows(W2),columns(W2)]


function w = packweights(W1,W2)
 w=[W1(:);W2(:)];
end

function [W1,W2] = unpackweights(w,rowsW1,colW1,rowsW2,colW2)
 W1=reshape(w(1:rowsW1*colW1),rowsW1,colW1);
 W2=reshape(w(rowsW1*colW1+1:end),rowsW2,colW2);
end

function y=gradientTarget(args)    
  w=args{1};
  wSize=args{2};
  X=args{3};
  Y=args{4};
  [W1,W2]= unpackweights(w,wSize(1),wSize(2),wSize(3),wSize(4));  
  y=target(W1,W2,X,Y);
endfunction;

function dy = dTarget(args)
  
  w=args{1};
  wSize=args{2};
  X=args{3};
  Y=args{4};
  [W1,W2]= unpackweights(w,wSize(1),wSize(2),wSize(3),wSize(4));

  dif=Y-predict(W1,W2,X);#difference between real and estimated value
  norma=(vectorNorm(dif)); #norm of each row 
  dy=sum(norma); #error funtion value
endfunction
w=packweights(W1,W2); #merging W1 and W2 into a single vector
errorIni=gradientTarget({w,wSize,X,Y});

[trainedW,minError,numEvaluations]=cg_min("gradientTarget","dTarget",{w,wSize,X,Y});
disp(" ")
disp("valor de error antes de optimizar:")
errorIni
disp(" ")
disp("numero de iteraciones/evaluaciones de gradientes conjugados:")
numEvaluations(1,1)
disp(" ");
disp("valor de error despues de optimizar:")
minError