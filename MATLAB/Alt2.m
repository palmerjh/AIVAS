function [ similarity ] = Alt2(a,b)
%GrayscaleAlt: Determines similarity of two grayscale images
%   1 - average absolute error
%   Reduces to XOR in BW case
error = 0.0;
size = numel(a);
for i = 1:size
    error = error + abs(a(i) - b(i));
end
similarity = 1.0 - error/size;

end