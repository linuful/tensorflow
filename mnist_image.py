from PIL import Image
from numpy import *

def RGB2Grey(R,G,B):
	return   (255-((R*30 + G*59 + B*11 + 50) / 100))


def GetImage(filename):
     #[batch, height, width, channels].
    width=28
    height=28
    value=zeros([1,width,height,1])
    value[0,0,0,0]=-1
    label=zeros([1,10])
    label[0,0]=-1

     #get image matrix
    img=array(Image.open(filename).convert("L"))
    width,height=shape(img);
    index=0
    tmp_value=zeros([1,width,height,1])
    for i in range(width):
        for j in range(height):
            tmp_value[0,i,j,0]=img[i,j]
            index+=1
    if(value[0,0,0,0]==-1):
        value=tmp_value
    else:
        value=concatenate((value,tmp_value))

    #get image label
    tmp_label=zeros([1,10])
    temp=filename.strip().split('/')
    last_idx=len(temp)-1
    index=int(temp[last_idx][0])
    #print("input:",filename)
    tmp_label[0,index]=1
    if(label[0,0]==-1):
        label=tmp_label
    else:
        label=concatenate((label,tmp_label))
    #return array(value),array(label)
    return value,label

#if __name__== '__main__':
    #img=array(Image.open("D:/2.png").convert("L"))
    #print((img))
    # GetImage(["images/1_2.png"])



