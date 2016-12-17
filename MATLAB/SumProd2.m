function [ similarity ] = SumProd2(a,b)
%GrayscaleSumProd: Determines similarity of two grayscale images
%   Uses ratio of pixelwise summations of prod to sum
%   prod reduces to intersection in BW case
%   sum reduces to XOR in BW case
product = 0.0;
summation = 0.0;

size = numel(a);
for i = 1:size
    product = product + a(i)*b(i);
    summation = summation + a(i)+b(i);
end

if (summation == 0)
    similarity = 0;
else
    similarity = 2 * product / summation;
end
end