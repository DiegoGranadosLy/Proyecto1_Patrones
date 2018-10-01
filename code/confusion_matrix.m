numElements=10000;
trainingNum=numElements*0.8;
[X,Y]=create_data(numElements,3,'pie'); #obtaining trainnig and test data
figure(1);
plot_data(X,Y); #plotting de training data
X=[ones(rows(X),1),X];  #bias
trainingX=X(1:trainingNum,1:end);
trainingY=Y(1:trainingNum,1:end);
testX=X(trainingNum:numElements,1:end);
testY=Y(trainingNum:numElements,1:end);
W1=[1,1,1;1,1,3;1,2,23]; #example values
W2=[1,2,2,3;1,6,3,1;1,2,3,5]; #example values

[W1,W2]=gradient_descent(W1,W2,trainingX,trainingY,100000); #training the neural network

classification=predict(W1,W2,testX)
#####################confusion matrix#######################
##confusion matrix template:
##                        Predicted classes
##                 class1      class2      class3   
##              ------------------------------------  
##  R C   class1|           |          |           |
##  e l         ----------------------------------     
##  a a   class2|           |          |           |
##  l s         ----------------------------------
##    s   class3|           |          |           |
##    e         ------------------------------------ 
##    s  
confusionMatrix=[0,0,0;0,0,0;0,0,0]; 
for i=1:rows(classification)
    [realValue,realClass]=max(testY(i,1:end)); #index of the maximun values represents its class
    [predictedValue,predictedClass]=max(classification(i,1:end));  #index of the maximun values represents its class
    confusionMatrix(realClass,predictedClass)++;

endfor
confusionMatrix
#########################################################
waitforbuttonpress();