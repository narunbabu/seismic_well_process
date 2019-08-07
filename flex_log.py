import numpy as np
import matplotlib.pyplot as plt
class FlexArray():
    #Input only a sorted array
    """
    This program is to find out the left and right value inside an array for a given values 
    """
    def __init__(self,sortedarray):
        self.flexarray=np.asarray(sortedarray) 
        self.stepsize=int(np.sqrt(self.flexarray.__len__()))
#         if self.flexarray.__len__()-1<self.stepsize:
#             self.stepsize=self.flexarray.__len__()-1

        self.flexarrIndexRangeArray=np.append(np.arange(0,len(self.flexarray)-1,self.stepsize),self.flexarray.__len__()-1)
    def get_stepsize(self):
        return self.stepsize
        
    def find_indx_nearest(self,array, value):
        array = np.asarray(array)
        return (np.abs(array - value)).argmin()    
        
    def find_indx_leftright(self,array,value):
        # print(value, array)
        # if value <= array[0]:
        #     print('yes yes')
        if (value>=array[-1]) | (value<=array[0]):
            if (value==array[-1]) | (value==array[0]):
                if (value==array[-1]):
                    return -2,array.__len__()-1
                else:
                    return 0,1
            print ('Value out of range!!')
            if (value>=array[-1]):
                return -1,np.nan
            else:
                return np.nan,0
#         array = 
        nearidx=(np.abs(array - value)).argmin()
        if value>array[nearidx]:
            return nearidx,nearidx+1
        elif value<array[nearidx]:
            return nearidx-1,nearidx
        else:
            return nearidx-1,nearidx+1
#         for i in range(self.flexarrIndexRangeArray.__len__()-1):
#             self.flexarray[self.flexarrIndexRangeArray[i]:self.flexarrIndexRangeArrayi+1]]
    def find_leftright_indxs(self,value):
        steparray=self.flexarray[self.flexarrIndexRangeArray]
        # print('step array : ',steparray)
        # print('val =',value)
        li,ri=self.find_indx_leftright(steparray,value)

        # if li==ri:
        #     return 0,0
#         print(li,ri,self.flexarrIndexRangeArray[[li,ri]])
        if (np.isnan(li))|(np.isnan(ri)):
            return li,ri
        else:
            smallarray=self.flexarray[self.flexarrIndexRangeArray[li]:self.flexarrIndexRangeArray[ri]+1]
        # print('hey',smallarray)
        try:
            lgi,rgi=self.find_indx_leftright(smallarray,value)
        except:
            print('---------------------------------bhoom---------------------------------------------')
            print(smallarray,value)
            return self.flexarrIndexRangeArray[li],self.flexarrIndexRangeArray[li]
        return self.flexarrIndexRangeArray[li]+lgi,self.flexarrIndexRangeArray[li]+rgi     

    def find_leftright_vlaues(self,value):
#         steparray=self.flexarray[self.flexarrIndexRangeArray]
# #         print(steparray)
#         li,ri=self.find_indx_leftright(steparray,value)
#         print(li,ri,self.flexarrIndexRangeArray[[li,ri]])
        
#         smallarray=self.flexarray[self.flexarrIndexRangeArray[li]:self.flexarrIndexRangeArray[ri]+1]
# #         print('hey',smallarray)
#         lgi,rgi=self.find_indx_leftright(smallarray,value)
#         return smallarray[[lgi,rgi]]
            if (value<self.flexarray[0]):
                return [np.nan,self.flexarray[0]]
            elif(value>self.flexarray[-1]):
                return [self.flexarray[-1],np.nan] 
            else:
                return self.flexarray[[self.find_leftright_indxs(value)]]
        
class FlexXY(FlexArray): 
    def __init__(self,XY):# XY should be an (N,2) shape np array
        self.XY=XY
        
        super(FlexXY,self).__init__(self.XY[:,0])
    def get_LRofXYs(self,xvalue):
        
        if (xvalue<self.XY[0,0]):
            return np.array([[self.XY[0,0]-self.stepsize,np.nan],[self.XY[0,0],self.XY[0,1]] ])
        elif(xvalue>self.XY[-1,0]):            
            return np.array([[self.XY[-1,0],self.XY[-1,1]],[self.XY[-1,0]+self.stepsize,np.nan] ])
        else:
            li,ri=self.find_leftright_indxs(xvalue)
    #         print(li,ri)
            return self.XY[[li,ri],:]
    def get_LRindxOfXYs(self,xvalue):
        li,ri=self.find_leftright_indxs(xvalue)
#         print(li,ri)
        return li,ri
    def predictYgivenX(self,xvalue):
        lr=self.get_LRofXYs(xvalue)
        return lr[0,1]+(xvalue-lr[0,0])*(lr[1,1]-lr[0,1])/(lr[1,0]-lr[0,0]) 

   
    def resampleY(self,new_xarray):
        # print(new_xarray)
        newY=np.zeros(len(new_xarray))
        for i in range(len(new_xarray)):
            newY[i]=self.predictYgivenX(new_xarray[i])
        return newY
    def resampleXY(self,new_xarray):
        newy=self.resampleY(new_xarray)#         
        return np.append(new_xarray.T,newy.T)


    def get_LRofMultiXYs(self,xvalue,xycols=[]): #xycols basically starts from 1  where actual x col value is absent. 
        # lenYs=len(self.XY[0,:])-1
        if len(xycols)==0:
            xycols=list(range(0,len(self.XY[0,:])))
        lenYs=len(xycols)-1
        # print('xycols ',xycols)
        # print('xvalue,self.XY[0,0],self.XY[-1,0] ',xvalue,self.XY[0,0],self.XY[-1,0])
        if (xvalue<self.XY[0,0]):
            return np.array([np.append(self.XY[0,0]-self.stepsize, [np.nan]*lenYs ),np.append(self.XY[0,0],self.XY[0,xycols]) ])
        elif(xvalue>self.XY[-1,0]):            
            return np.array([np.append(self.XY[-1,0],self.XY[-1,xycols]),np.append(self.XY[-1,0]+self.stepsize,[np.nan]*lenYs) ])
        else:
            li,ri=self.find_leftright_indxs(xvalue)
            # print('li,ri ', li,ri)
            # print(np.array(self.XY),np.array(self.XY)[[li,ri],xycols] )
            # resXY=[]
            # for xyc in xycols:
            #     resXY.append(self.XY[[li,ri],xyc])
            # print(self.XY[li,xycols],'\n',self.XY[ri,xycols])
            return np.vstack((self.XY[li,xycols],self.XY[ri,xycols]))
            # print('resXY',resXY)
            # return np.nan
            # return np.array(self.XY)[[li,ri],xycols] 
    def predictYgivenXYs(self,xvalue,ycol=1):
        lr=self.get_LRofMultiXYs(xvalue,xycols=[0,ycol])#xycols really matters here
        # print(lr)
        return lr[0,1]+(xvalue-lr[0,0])*(lr[1,1]-lr[0,1])/(lr[1,0]-lr[0,0])#xycols already considered in above statement
       
    def predictYsgivenX(self,xvalue,xycols=[]):
        if len(xycols)==0:
            xycols=list(range(0,len(self.XY[0,:])))
        lr=self.get_LRofMultiXYs(xvalue,xycols=xycols)#xycols really matters here
        # print(lr)
        return lr[0,1:]+(xvalue-lr[0,0])*(lr[1,1:]-lr[0,1:])/(lr[1,0]-lr[0,0])#xycols already considered in above statement
    def predictYsgivenXavg(self,xvalue,xycols=[]):
        if len(xycols)==0:
            xycols=list(range(0,len(self.XY[0,:])))
        lr=self.get_LRofMultiXYs(xvalue,xycols=xycols)#xycols really matters here
        # print(lr)
        return lr[0,1:]+(xvalue-lr[0,0])*(lr[1,1:]-lr[0,1:])/(lr[1,0]-lr[0,0])#xycols already considered in above statement
    def resampleYs(self,new_xarray,xycols=[]):
        import matplotlib.pyplot as plt
        # print(new_xarray)
        if len(xycols)==0:
            xycols=list(range(0,len(self.XY[0,:]))) #if xycols not specified all y cols will be taken
        lenYs=len(xycols)-1
        # print('Length of ys ',lenYs)
        newYs=np.zeros((len(new_xarray),lenYs))
        for i in range(len(new_xarray)):
            # print(new_xarray[i],self.XY[0,0],end=', ')
            # lenYs=len(self.XY[0,:])-1
            if(new_xarray[i]<self.XY[0,0]) | (new_xarray[i]>self.XY[-1,0]):
                newYs[i,:]=[np.nan]*lenYs
            else:
                # ys=self.predictYsgivenX(new_xarray[i])
                # print(ys)
                if(len(xycols)>1):
                    newYs[i,:]=self.predictYsgivenX(new_xarray[i],xycols=xycols)
                else:
                    newYs[i,:]=self.predictYgivenXYs(new_xarray[i],ycol=xycols[0])
        # plt.plot(new_xarray,newYs)
        # print('Paused..............')
        # plt.pause(5)
        
        # plt.show()
        return newYs
    def resampleXYs(self,new_xarray):
        newy=self.resampleYs(new_xarray)#         
        return np.column_stack((new_xarray,newy))
class FlexLog(FlexXY):
    def __init__(self,XY):
        self.XY=XY
        super(FlexLog,self).__init__(self.XY)
        self.calc_logstep()
        # self.logstep=self.XY[0,0]-self.XY[:,0]
        # np.unique(np.diff(self.XY[:,0].T))        
        # if self.logstep.__len__()>1:
        #     print( 'log step size varies... not a log array')
        #     return None
    def clipleftnans(self,x,y):
        prev=True
        for i in range(len(y)):
            if prev & ~np.isnan(y[i]):        
                return x[i:],y[i:]
    def cliprightnans(self,x,y):
        x,y=x[::-1],y[::-1]
        x,y=self.clipleftnans(x,y)
        return x[::-1],y[::-1]
    def clipedgenans(self,x,y):    
        return self.cliprightnans(*self.clipleftnans(x,y)) 
    def get_nonanXYs(self,x,y):
        x,y=self.clipedgenans(x,y)
        myflexlog= FlexLog(np.column_stack((x,y)))
        nanindxs=np.isnan(y)
        myflexlog.XY=myflexlog.XY[~nanindxs,:]
        # return np.where(nanindxs)[0],myflexlog
        for nanind in np.where(nanindxs)[0]:
            xvalue=x[nanind]
            lridxs=myflexlog.find_indx_leftright(myflexlog.XY[:,0],xvalue)
            lr=myflexlog.XY[lridxs,:]
            y[nanind]=lr[0,1:]+(xvalue-lr[0,0])*(lr[1,1:]-lr[0,1:])/(lr[1,0]-lr[0,0])
        # for nanind in np.where(nanindxs)[0]:
        #     ys=myflexlog.get_LRofXYs(xvalue)
        #     # y[nanind]=myflexlog.predictYsgivenX(x[nanind])
        return x,y
    def calc_logstep(self):
        for i in range(1,len(self.XY[:,0])):
            self.logstep=np.abs(self.XY[i,0]-self.XY[i-1,0])
            if not np.isnan(self.logstep):
                break
    def logExtend(self,newLog,depthminmax=[None,None],replace='top'):
#         newLog=FlexLog(newLog)
        # print('**********************************************************************')
        # print(self.XY[:,0])
        # plt.plot(self.XY[:,0],self.XY[:,1:])
        # plt.title('xy in logextend start.')
        # plt.pause(6)
        # plt.show()
        # plt.close()

        # plt.plot(newLog[:,0],newLog[:,1:])
        # plt.title('newlog in logextend start.')
        # plt.pause(6)
        # plt.show()
        # plt.close()
        # print(len(self.XY[:,0]))
        if len(self.XY[:,0])<2:
            super(FlexLog,self).__init__(newLog)

            return
        else:
            # self.logstep=self.XY[1,0]-self.XY[0,0]
            self.calc_logstep()
        self.toplog='existing'
        # print('new log and olg log depths',newLog[0,0],newLog[-1,0],self.XY[0,0],self.XY[-1,0])
        # print(newLog[-1,0]>self.XY[-1,0])
        # if not depthminmax[0]:
        # print('it is here...........................................')
        if(newLog[0,0]<self.XY[0,0]):
            depthminmax[0]=newLog[0,0]
            self.toplog='incoming'
        else:                
            depthminmax[0]=self.XY[0,0]
        if(newLog[-1,0]>self.XY[-1,0]):
            depthminmax[1]=newLog[-1,0]
        else:
            depthminmax[1]=self.XY[-1,0]
        flexnewLog=FlexLog(newLog)
        # print('depthminmax ',depthminmax,self.logstep)
        merged_xarray=np.arange(depthminmax[0],depthminmax[1],self.logstep)
        if self.toplog=='existing':
            if replace=='top':
                start_depth_bottom_log=newLog[0,0]
            else:
                start_depth_bottom_log=self.XY[-1,0]+self.logstep
        else:
            if replace=='top':
                start_depth_bottom_log=self.XY[0,0]
            else:
                start_depth_bottom_log=newLog[-1,0]+self.logstep

        # print('start_depth_bottom_log: ',start_depth_bottom_log)
        # print('Top log xys: ',self.XY[-1,:])
        flexmdarray=FlexArray(merged_xarray)

        lrindexes=flexmdarray.find_leftright_indxs(start_depth_bottom_log)
        # 
        if not (np.isnan(np.sum(lrindexes))):
            bot_log_darray=merged_xarray[lrindexes[1]:]
            top_log_limitIndex=np.where(self.XY[:,0]==merged_xarray[lrindexes[0]])[0]
        else:
            idx=self.find_indx_nearest(merged_xarray, start_depth_bottom_log)
            bot_log_darray=merged_xarray[idx:]
            top_log_limitIndex=[idx-1]
        # print(bot_log_darray)    
        # print(top_log_limitIndex)
        # print('lrindexes ',lrindexes)
        if len(top_log_limitIndex)==0:
            top_log_darray=merged_xarray[:lrindexes[0]]

            oldYs=self.resampleYs(top_log_darray)
            
        else:
            top_log_darray=merged_xarray[:top_log_limitIndex[0]]
            oldYs=self.XY[:top_log_limitIndex[0],1:]
        # print('lrindexes ',lrindexes)
        # print('top_log_darray top and end ',top_log_darray[0],top_log_darray[-1])
        # print('bot_log_darray first and last ',bot_log_darray[0], bot_log_darray[-1])   
        
        newYs=flexnewLog.resampleYs(bot_log_darray)
        # print('size newYs',newYs.shape)
        # print('len x array',len(merged_xarray))
        self.XY=np.column_stack((np.append(top_log_darray,bot_log_darray),np.append(oldYs,newYs,axis=0)))
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print(oldYs,newYs)
        # # print(self.XY[:,:])
        # plt.plot(self.XY[:,0],self.XY[:,1:])
        # plt.title('to see xy in flexoldlog okey...')
        # plt.pause(6)
        # plt.show()
        # plt.close()
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
        super(FlexLog,self).__init__(self.XY)
        # print(self.XY.shape)

    def getSplicedLog(self,logstep=0.1524):
        # exist_logstep=self.XY[1,0]-self.XY[0,0]
        # print('--------------------------in getSplicedLog')
        if self.logstep != logstep:
            # print('--------------------------in before resample')
            # plt.plot(self.XY[:,0],self.XY[:,1:])
            # plt.pause(10)
            
            new_d=np.arange(self.XY[0,0],self.XY[-1,0],logstep)
            # print('--------------------------in after resample')
            resXY=self.resampleXYs(new_d)
            # plt.plot(new_d,resXY[:,1:])
            # plt.pause(10)
            # plt.cla()
            return resXY
        else:     
            # print('--------------------------in getSplicedLog else paart..........')    
            return self.XY

    def clip(self,drange=(None,None)):
        self.XY=self.XY[(self.XY[:,0]>=drange[0]) & (self.XY[:,0]<=drange[1])]
   

class FlexLogCurves(FlexLog):
    def __init__(self,XY=np.array([[0,0]]),YKeys=[]):
        # self.XY=XY
        # print(self.XY[:,0])
        super(FlexLogCurves,self).__init__(XY)
        self.YKeys=np.array(YKeys)
        # print(YKeys)
    def sameSuitAppend(self,newLog,newYKeys,depthminmax=[None,None],replace='top'):
    #         newLog=FlexLog(newLog)
        # print('newYKeys',newYKeys)
        # print('sameSuitAppend**********************************************************************')
        
        # print(len(self.XY[:,0]))
        if len(self.XY[:,0])<2:
            super(FlexLog,self).__init__(newLog)
            # print(self.XY.shape)
            return
        else:
            if not hasattr(self, 'logstep'):
                for i in range(1,len(self.XY[:,0])):
                    self.logstep=np.abs(self.XY[i,0]-self.XY[i-1,0])
                    if not np.isnan(self.logstep):
                        break
            
            # print('self.logstep',self.logstep)
        self.toplog='existing'
        xymin,xymax,newlogmin,newlogmax = min(self.XY[:,0]),max(self.XY[:,0]),min(newLog[:,0]),max(newLog[:,0])
        # print('newlogmin,newlogmax',newlogmin,newlogmax)
        # print('xymin,xymax',xymin,xymax)
        if(newlogmin<xymin):
            depthminmax[0]=newlogmin
            self.toplog='incoming'
        else:                
            depthminmax[0]=xymin
        if(newlogmax>xymax):
            depthminmax[1]=newlogmax
        else:
            depthminmax[1]=xymax
        flexnewLog=FlexLogCurves(newLog,newYKeys)
        # print('depthminmax self.logstep, ',depthminmax,self.logstep)

        merged_xarray=np.arange(depthminmax[0],depthminmax[1],self.logstep)


        # print('start_depth_bottom_log: ',start_depth_bottom_log)
        # print('Top log xys: ',self.XY[-1,:])
        flexmdarray=FlexArray(merged_xarray)

        lrindexes=flexmdarray.find_leftright_indxs(depthminmax[0])
        # plt.plot(merged_xarray,oldYs)
        # plt.show()
        # plt.plot(merged_xarray,newYs)
        # plt.show()
        # print(merged_xarray,self.XY[:,0])
        # plt.plot(self.XY[:,0],self.XY[:,1:])
        # plt.show()
        # print(merged_xarray,flexnewLog.XY[:,0])
        # plt.plot(flexnewLog.XY[:,0],flexnewLog.XY[:,1:])
        # plt.show()
        # if (xymin!=depthminmax[0])|(xymax!=depthminmax[1]): 
        oldYs=self.resampleYs(merged_xarray) 
        # else:           
        # if (newlogmin!=depthminmax[0])|(newlogmax!=depthminmax[1]):
        newYs=flexnewLog.resampleYs(merged_xarray)

        
        # print('size newYs',newYs.shape,'size oldys ',oldYs.shape)
        # print('len x array',len(merged_xarray))
        self.XY=np.column_stack((merged_xarray,np.append(oldYs,newYs,axis=1)))
        self.YKeys=np.append(self.YKeys,newYKeys)
        # print('self.YKeys:  ', self.YKeys)
        # plt.plot(self.XY[:,0],self.XY[:,1:])
        # plt.show()
        self.__init__(self.XY,self.YKeys)
        # print(self.XY.shape)

         
if __name__=='__main__':
    np.set_printoptions(edgeitems=3,infstr='inf',linewidth=75, nanstr='nan', precision=2,suppress=False, threshold=1000, formatter=None)

    a=np.arange(0,20,0.05)
    a.shape=len(a),1
    c=np.arange(0,20,0.1)
    b=np.sin(a)
    xy=np.append(a,b,axis=1)
    plt.plot(xy[:,0],xy[:,1],'-*')
    # plt.plot(a,b)
    flexXY=FlexLog(xy)

    newxy=flexXY.resampleXY(c)

    plt.plot(newxy[:,0],newxy[:,1])
    plt.show()
