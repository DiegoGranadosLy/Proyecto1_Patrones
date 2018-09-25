
function w = packweights(W1,W2)
 w=[W1(:);W2(:)]
end

function [W1,W2] = unpackweights(w,rowsW1,colW1,rowsW2,colW2)
 W1=reshape(w(1:rowsW1*colW1),rowsW1,colW1)
 W2=reshape(w(rowsW1*colW1+1:end),rowsW2,colW2)
end
