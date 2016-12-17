function [ similarity ] = MaxMin2(a,b)
%GrayscaleMaxMin: Determines similarity of two grayscale images
%   Uses ratio of pixelwise summations of min to max
%   min reduces to intersection in BW case
%   max reduces to union in BW case
minimum = 0.0;
maximum = 0.0;

size = numel(a);
for i = 1:size
    minimum = minimum + min(a(i),b(i));
    maximum = maximum + max(a(i),b(i));
end

if (maximum == 0)
    similarity = 1;
else
    similarity = minimum / maximum;
end
end



