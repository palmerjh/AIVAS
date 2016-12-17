function [ results ] = Search( search_im,target,locations,queueSize )
% locations is a list of "boxes": 4-tuples of form (x0,y0,xn+1,yn+1)

t1 = rgb2gray(imread('target1.jpg'));
t1 = 1 - im2double(t1);

    most_sim = {'sp':[],   # format: [(most_sim, (pixel))]
                'mm':[],
                'alt':[]}

    tx, ty = target.size()  
    subSpace = np.empty((tx,ty))

    '''
    ssx, ssy = searchSpace.size()
    spArr = np.empty((ssx-tx+1,ssy-ty+1))
    altArr = np.empty((ssx-tx+1,ssy-ty+1))
    '''
    
    nIterations = 0
    for box in locations:
        nIterations += (box[2] - box[0]) * (box[3] - box[1])
        
    print nIterations

    counter = 0
    for box in locations:
        for sx in xrange(box[0],box[2]):
            for sy in xrange(box[1],box[3]):
                subSpace = np.array(searchSpace.pix[sx:sx+tx,sy:sy+ty])
                
                sp = sim.sumProd(target.pix,subSpace)
                mm = sim.maxMin(target.pix,subSpace)
                a = sim.alt(target.pix,subSpace)

                '''
                if queueSize == 1:
                    print('\n%d:\tsp %f.3, mm %f.3, alt %f.3 at %s' % (counter,sp,mm,a,str((sx,sy))))
                    print('before:')
                    if len(most_sim['sp']) == 0:
                        print('sp:\nmm:\nalt:')
                    else:
                        for key in most_sim.keys():
                            print('%s:\t%s' % (key,str(most_sim[key])))
                              
                if not queueSize == 1:
                    spArr[sx,sy] = sp
                    altArr[sx,sy] = a
                

                if counter < queueSize:
                    hq.heappush(most_sim['sp'],(sp,(sx,sy)))
                    hq.heappush(most_sim['mm'],(mm,(sx,sy)))
                    hq.heappush(most_sim['alt'],(a,(sx,sy)))
                else:
                    hq.heappushpop(most_sim['sp'],(sp,(sx,sy)))
                    hq.heappushpop(most_sim['mm'],(mm,(sx,sy)))
                    hq.heappushpop(most_sim['alt'],(a,(sx,sy)))

                
                if queueSize == 1:
                    print('\nafter:')
                    for key in most_sim.keys():
                        #print('%s:\t%f at %s' % (key,round(most_sim[key][0],3),str(most_sim[key][1])))
                        print('%s:\t%s' % (key,str(most_sim[key])))
                '''  
                

                if counter % 1000 == 0:
                    print(str(round(100*(float(counter)/nIterations),2)) + r'% done')
                    
                if counter == 10000:
                    print 'done'

                #print(str(counter) + ': ' + str(most_sim['sp']))

                counter += 1

    for key in target.most_sim.keys():
        #print(most_sim[key])
        target.most_sim[key] = [tup[1] for tup in most_sim[key]]

    '''
    if not queueSize == 1:
        debug(spArr,'sp'+target.fname.split('.')[0])
        debug(altArr,'alt'+target.fname.split('.')[0])
    '''

    return most_sim if queueSize == 1 else None


end

