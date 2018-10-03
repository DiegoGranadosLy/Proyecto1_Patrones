[X,Y]=create_data(10000,3,'pie'); #obtaining trainnig data
figure(1);
plot_data(X,Y); #plotting de training data
X=[ones(rows(X),1),X];  #bias
W1=[1,1,1;1,1,3;1,2,23]; #example values
W2=[1,2,2,3;1,6,3,1;1,2,3,5]; #example values

##this function trains the neural network for a given set of data X and it set of labels Y
## then it classifies all the points of a image using the trained neural network
function v = visualization(W1,W2,X,Y)
    k=3; #number of classes 
    [W1,W2]=gradient_descent(W1,W2,X,Y,10000); #training the neural network
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

end

visualization(W1,W2,X,Y);
waitforbuttonpress ();