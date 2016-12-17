function [ similarity ] = MaxMin(a,b)
%GrayscaleMaxMin: Determines similarity of two grayscale images
%   Uses ratio of pixelwise summations of min to max
%   min reduces to intersection in BW case
%   max reduces to union in BW case
minimum = sum(sum(min(a,b)));
maximum = sum(sum(max(a,b)));

if (maximum == 0)
    similarity = 1;
else
    similarity = minimum / maximum;
end
end



