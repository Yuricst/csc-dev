"""
bepicolombo_cosmogen.py
Visualize Earth swing-by of BepiColombo using JPL's Cosmographia
This script generates the catalog file (.JSON) to be opened in Cosmographia
Yuri Shimane, 2020/04/13
"""

# add to path location of cosmonout library
import sys
sys.path.append("../../cosmonout")
# import cosmonout library
import cosmonout

# define path to local Cosmographia installation
pathtoCosmoInstal = '/home/yuri/apps/cosmographia-4.1'


# ----- KERNEL FILES AND SPACECRAFT MODEL OF TRAJECTORY -----
# bsp (trajectory) kernel
bspKernel = '../notebooks/spice/baseline_outbound_SE.bsp'
# # frame kernel
# myCKernel = './spice/bc_mpo_sc_fmp_EarthSwingbyMTP_00001_f20181127_v01.bc'
# # spacecraft clock kernel
# mySCLKernel = './spice/bc_mpo_step_20200325.tsc'

# provide asbolute path to spacecraft 3D model
myModel = None #'./models/bepi_mcs.3ds'


# ----- DEFINE LOCATIONS TO SAVE FILES THAT WILL BE GENERATED -----
# define absolute to save catalogs
mycats = './cosmographia_catalogs'

# define location so save cosmoscripts
myscript = './cosmoscripts'


# ----- CALL COSMONOUT FUNCTIONS -----
# generate frame kernel, catalog files, state URL
trajvis = cosmonout.cosmoPrep(pathtoCosmoInstal, bspKernel, 
	ckernel=None, sclkernel=None, catdir=mycats, modelpath=myModel, 
	scname='TheORACLE', scriptdir=None, interpts=10000
)

# launch Cosmographia with catalog and state URL from trajvis1
cosmonout.cosmoLaunch(trajvis)


