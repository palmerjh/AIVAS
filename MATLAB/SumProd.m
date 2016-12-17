function [ similarity ] = SumProd(a,b)
%GrayscaleSumProd: Determines similarity of two grayscale images
%   Uses ratio of pixelwise summations of prod to sum
%   prod reduces to intersection in BW case
%   sum reduces to XOR in BW case
product = sum(sum(a.*b));
summation = sum(sum(a+b));

if (summation == 0)
    similarity = 0;
else
    similarity = 2 * product / summation;
end
end