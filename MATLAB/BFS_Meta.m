search_im = rgb2gray(imread('search.jpg'));
t1 = rgb2gray(imread('target1.jpg'));
t2 = rgb2gray(imread('target2.jpg'));
rf = 0.25; % resize factor
qs = 200; % priority queue size

search_im_small = imresize(search_im,rf);
t1_small = imresize(t1,rf);
t2_small = imresize(t2,rf);



