numElements=10000; #the 80% of this data is for training and the other 20% is for testing 
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

classification=predict(W1,W2,testX);#clasification using test data


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
tp1=confusionMatrix(1,1);
tp2=confusionMatrix(2,2);
tp3=confusionMatrix(3,3);
fp1=sum(confusionMatrix(2:3,1));
fp2=confusionMatrix(1,2)+confusionMatrix(3,2);
fp3=confusionMatrix(1,3)+confusionMatrix(2,3);
disp("la matriz de confusión obtenida es:")
disp(confusionMatrix);
###sesitivity for each of the classes
disp(" ") 
disp("sensibilidad para la clase 1:");
sensitivity1=tp1/sum(confusionMatrix(1,1:end))
disp("sensibilidad para la clase 2:");
sensitivity2=tp2/sum(confusionMatrix(2,1:end))
disp("sensibilidad para la clase 3:");
sensitivity3=tp3/sum(confusionMatrix(3,1:end))
###precision for each of the classes
disp(" ")
disp("precisión para la clase 1:");
precision1=tp1/(tp1+fp1)
disp("precisión para la clase 2:");
precision2=tp2/(tp2+fp2)
disp("precisión para la clase 3:");
precision3=tp3/(tp3+fp3)
###F1 score for each of the classes
disp(" ")
disp("valor F1 para la clase 1:");
Fscore1=(2*sensitivity1*precision1)/(sensitivity1+precision1)
disp("valor F1 para la clase 2:");
Fscore2=(2*sensitivity2*precision2)/(sensitivity2+precision2)
disp("valor F1 para la clase 3:");
Fscore3=(2*sensitivity3*precision3)/(sensitivity3+precision3)
#########################################################
waitforbuttonpress();