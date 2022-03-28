#
# Trajectory visualization complementary cosmoscript
# Automatically generated
#

import cosmoscripting
cosmo = cosmoscripting.Cosmo()

cosmo.displayNote('Fast forward to end of trajectory...', 5).wait(3)
cosmo.setTimeRate(1000000)
cosmo.showTrajectory('TheORACLE')
cosmo.pause()
cosmo.showTrajectory('TheORACLE')
cosmo.setTimeRate(1)
