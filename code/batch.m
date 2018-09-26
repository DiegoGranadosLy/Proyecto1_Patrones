
function [nX,nY] = batch(X,Y,batchSize)
  nX=[];
  nY=[];
  if(batchSize<rows(X))
    for i= 1:batchSize
      r=randi([1,rows(X)]);
      nX=[X(r,:);nX];
      nY=[Y(r,:);nY];
    endfor
  else
    nX=X;
    nY=Y;
  endif
end