function [ similarity ] = BWSimilarity1(image1,image2)
%BWSimilarity: Determines similarity of two black and white images
%   Uses ratio of intersection to union
i = 0;
u = 0;
for n = 1:numel(image1)
    if(image1(n) == 1 && image2(n) == 1)
        i = i + 1;
    end
    if(image1(n) == 1 || image2(n) == 1)
        u = u + 1;
    end
end
if(u == 0)
    similarity = 0;
else
    similarity = i./u;
end
end

