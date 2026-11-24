import numpy as np
from pmwMRMS.archive import phase_rates

def test_phase_rates():
    r,s=phase_rates(np.ones(5),np.array([0,1,3,7,-3]))
    np.testing.assert_allclose(r,[0,1,0,np.nan,np.nan])
    np.testing.assert_allclose(s,[0,0,1,np.nan,np.nan])
