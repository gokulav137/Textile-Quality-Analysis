'''  1.FINDING DISTANCE BETWEEN TWO WHITE BLOBS TO DECREASE RUNTIME  '''

count = np.zeros(2*n)  #empty arrays to store distances
a,b = image.shape  # a - rows, b - columns
n = 30#no of test samples
for i in range(0,a-1,int(a/n)):   #iterate over random n rows seperated by a/n distance
    e =int( i/int(a/n) )   # index of a row(0 - n-1)
    
    for x in range(int(b)):  # looping through ith row
        if filter[i,x]==255: #finding first white blob
            for y in range(x,int(b/10)):
                if filter[y,i] == 0:  #tracking y as the first black pix between two white blobs
                    break
                else:
                    continue
                  
        else:
            continue
    
   # print(i,e,k)
    for j in range(y,b-1):
       
        if filter[i,j]!=255 :
           # k = filter[i,j+1]
                count[e] = count[e]+1  
        else :
                #print(i,count[e])
                break
 
    
count = count[(count>0)]  
#count= np.sort(count,axis=0)  
#t1 = datetime.now() - t0 
#len = np.sum(count) / n

''' 2. TO AN IMAGE WITH CONNECTED BLOBS IN A ROW ,THIS COUNTS THE WEFTS'''

weftcount = 0
draw_weft = 1

if draw_weft:      
    im2y, contoursy, hierarchyy = cv2.findContours(threshvis,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) 
    for cnt in contoursy:        
      #  perimeter = cv2.arcLength(cnt,True) 
      #  if perimeter > weft_perimeter_thresh:          
            cv2.drawContours(img, cnt, -1, (0,0,255), 1)          
            weftcount = weftcount+1          
            #weftarea = +cv2.contourArea(cnt)
            
            
            
            
''' 3. TO DILATE AN IMAGE HORIZONTALLY ,KERNEL'S LIKE THIS COULD BE USEFUL'''

kernel1 = np.array([[0,0,0,0,0,0,0,0,0,0],[10,10,10,10,10,10,10,10,10,10],[0,0,0,0,0,0,0,0,0,0]],np.uint8)
filtered_img = cv2.filter2D(filtered_thresh_img, cv2.CV_8UC3,kernel1)




''' 4.SUM OF ROWS METHOD '''


a,b = threshvis.shape         

row_sum = np.zeros((a,1))
for i in range(a):
    row_sum[i] = np.sum(threshvis[i])/b #TAKES AVG OF ALL ROWS
    print(row_sum[i]) 
for i in range(a):
    if row_sum[i] <np.sum(row_sum)/a:  #IF A ROW'S SUM IS LESS THAN AVG, ITS MADE TO 0'S 
        threshvis[i] = 0
    else :
        threshvis[i] = 255