function [ similarity ] = BWSimilarity2(image1,image2)
%BWSimilarity: Determines similarity of two black and white images
%   Uses ratio of intersection to union
i = sum(sum(image1 & image2));
u = sum(sum(image1 | image2));
if(u == 0)
    similarity = 0;
else
    similarity = i./u;
end
end

