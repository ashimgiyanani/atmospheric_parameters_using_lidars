import pandas as pd
from pathlib import Path
import glob
import matlab2py as m2p

raw_files = r"C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\atmospheric_param_lidars\data\Tauern\202105\20210504"
files = glob.glob(raw_files + "\*.hpl")
print(files)


# # read in a hpl file
import os
from io import BufferedReader, BytesIO
from haloreader.exceptions import FileEmpty, UnexpectedDataTokens
from haloreader.read import _read_single, read, read_bg, _read_header, _read_data
src=Path(raw_files).joinpath(os.path.basename(files[0]))
header_end, header_bytes = _read_header(src)
df_halo =  _read_single(src)

# get the mean beta for all ranges
df_beta = pd.DataFrame(df_halo.beta_raw.data, columns=df_halo.range.data, index=df_halo.time.data)
profile_beta = df_beta.mean(axis=0)


#%% find the first zero crossing of beta derivative
import numpy as np
idx_zero_cross = np.where(profile_beta.diff().apply(np.signbit))[0][0]
dbeta = np.diff(profile_beta.values)
# plt.plot(dbeta[0:9], profile_beta.index[0:9])
# plt.xticks(rotation=45)

# plot the beta distribution
import matplotlib.pyplot as plt
plt.plot(profile_beta.values[0:15], profile_beta.index[0:15], 'b.-', label='beta profile')
plt.plot(profile_beta.values[idx_zero_cross], profile_beta.index[idx_zero_cross], 'ro', label='cloud height range')
plt.plot(profile_beta.diff().iloc[0:15],profile_beta.index[0:15], 'g:', label='beta gradient')
plt.plot(profile_beta.diff().iloc[idx_zero_cross], profile_beta.index[idx_zero_cross], 'ro', label='cloud height range')
plt.legend()
plt.show()

#%% similarly for intensity values
df_intensity = pd.DataFrame(df_halo.intensity_raw.data, columns=df_halo.range.data, index=df_halo.time.data)
profile_intensity = df_intensity.mean(axis=0)
# find the first zero crossing of intensity derivative
import numpy as np
idx_zero_cross = np.where(profile_intensity.diff().apply(np.signbit))[0][0]
dintensity = np.diff(profile_intensity.values)
# plt.plot(dintensity[0:9], profile_intensity.index[0:9])
# plt.xticks(rotation=45)

# plot the intensity distribution

import matplotlib.pyplot as plt
plt.plot(profile_intensity.values, profile_intensity.index, 'b-', label='intensity profile')
plt.plot(profile_intensity.values[idx_zero_cross], profile_intensity.index[idx_zero_cross], 'ro', label='cloud height range')
plt.plot(profile_intensity.diff(),profile_intensity.index, 'g:', label='intensity gradient')
plt.plot(profile_intensity.diff().iloc[idx_zero_cross], profile_intensity.index[idx_zero_cross], 'ro', label='cloud height range')
plt.legend()
plt.show()
