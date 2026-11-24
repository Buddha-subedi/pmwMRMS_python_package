import numpy as np
from pmwMRMS.orbits import assignments, aggregate, nearest_time

def test_average_and_time():
    a=dict(lat=np.array([[40.,40.]]),lon=np.array([[-100.,-99.]]),time=np.array([[100.,230.]]),data=np.ones((1,2,1))*200)
    _,owner,t=next(assignments(a,np.array([40.]),np.array([-100.,-99.99,-99.]),5,np.array([100.,220.]),2))
    np.testing.assert_array_equal(owner,[[0,0,1]])
    np.testing.assert_array_equal(t,[[0,0,1]])
    c,r,s=aggregate(owner,np.array([[0.,6.,2.]]),np.array([[0.,0.,1.]]),2)
    np.testing.assert_allclose(r/c,[3,2]); np.testing.assert_allclose(s/c,[0,1])

def test_missing():
    c,r,s=aggregate(np.array([0,0,-1]),np.array([2.,np.nan,3.]),np.array([0.,0.,0.]),1)
    assert c[0]==1 and r[0]==2
