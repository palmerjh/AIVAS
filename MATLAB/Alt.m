function [ similarity ] = Alt(a,b)
%GrayscaleAlt: Determines similarity of two grayscale images
%   1 - average absolute error
%   Reduces to XOR in BW case

size = numel(a);
similarity = 1.0 - sum(sum(abs(a-b)))/size;

end