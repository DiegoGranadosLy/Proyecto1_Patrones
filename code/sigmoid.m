#sigmoid funtion that works for scalars and matrixes
function g = sigmoid(X)
  g = 1 ./ (1 + e.^-X);
end