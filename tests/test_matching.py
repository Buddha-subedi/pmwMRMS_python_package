import numpy as np
from pmwMRMS.automatic import map_group

def test_time_and_distance():
    src=dict(lat=np.array([[40.,40.]]),lon=np.array([[-100.,-99.]]),time=np.array([[100.,1000.]]),data=np.array([[[200.],[250.]]]))
    tb,d,dt=map_group(src,np.array([40.]),np.array([-100.,-99.]),100.,7.5,15.)
    assert tb[0,0,0]==200 and np.isnan(tb[0,1,0])
    assert d[0,0,0]<1e-5 and dt[0,0,0]==0
