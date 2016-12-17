x = [1 2 3 4 5 6 7 8 9 10];
y1 = [.16 .08 .04 .02 .013 .007 .004 .002 .001 .0008 ];
y2 = [.16 .07 .03 .01 .008 .003 .0008 .0003 .00007 .00002 ];

%semilogy(x,y1,'-bo;y1;',x,y2,'-kx;y2;');
title('Plot title');
xlabel('X Axis');
ylabel('Y Axis');
%print -deps graph.eps