'''
Author: Jia-Lai Kang

Using simulation to derive the correction factors for the biases in the flux-resovled X-ray spectroscopy.

Requires:
    numpy, astropy
'''

import numpy as np
from astropy.io import fits
import sys

# assert that A and B modules have the same background scale
def perfrom_one_simulate(files, key_count_rate, simulation_number=10000):
    # obtain the scales of the background and source regions from the spectra files
    spectra_source_A = fits.open(files[4]);
    spectra_back_A = fits.open(files[5])
    backscal_factor = spectra_source_A[1].header['backscal'] / spectra_back_A[1].header['backscal']
    spectra_source_A.close();
    spectra_back_A.close()

    # open the light curves for FPMA and FPMB, in source and background regions
    source_A = fits.open(files[0]);
    source_B = fits.open(files[1])
    back_A = fits.open(files[2]);
    back_B = fits.open(files[3])

    timebin = source_A[1].data['time'][1] - source_A[1].data['time'][0]

    data_source = []
    data_back = []

    # drop the time bins with exposure fraction (fracexp) < fracexp_cut
    fracexp_cut = 0.9
    locations_A = np.where(source_A[1].data['fracexp'] > fracexp_cut)
    source_A[1].data = source_A[1].data[locations_A]
    back_A[1].data = back_A[1].data[locations_A]
    
    locations_B = np.where(source_B[1].data['fracexp'] > fracexp_cut)
    source_B[1].data = source_B[1].data[locations_B]
    back_B[1].data = back_B[1].data[locations_B]

    # In some cases, the Good Time Intevals of FPMA and FPMB can be different.
    # Here we align two lcs and caculate the sum of A and B
    cursor = 0
    for i in range(len(source_A[1].data['time'])):
        for j in range(cursor, len(source_B[1].data['time'])):
            if source_A[1].data['time'][i] == source_B[1].data['time'][j]:
                itemA = source_A[1].data[i];
                itemB = source_B[1].data[j]
                total_rate = (itemA['rate'] * itemA['fracexp'] + itemB['rate'] * itemB['fracexp']) / (
                        itemB['fracexp'] + itemA['fracexp']) * 2
                total_rate_origin = (itemA['rate_orig'] * itemA['fracexp'] + itemB['rate_orig'] * itemB['fracexp']) / (
                        itemB['fracexp'] + itemA['fracexp']) * 2
                data_source.append(
                    [itemA['time'], total_rate, total_rate_origin, (itemB['fracexp'] + itemA['fracexp']) / 2])
                itemA = back_A[1].data[i];
                itemB = back_B[1].data[j]
                total_rate = (itemA['rate'] * itemA['fracexp'] + itemB['rate'] * itemB['fracexp']) / (
                        itemB['fracexp'] + itemA['fracexp']) * 2
                total_rate_origin = (itemA['rate_orig'] * itemA['fracexp'] + itemB['rate_orig'] * itemB['fracexp']) / (
                        itemB['fracexp'] + itemA['fracexp']) * 2
                data_back.append(
                    [itemA['time'], total_rate, total_rate_origin, (itemB['fracexp'] + itemA['fracexp']) / 2])

                cursor = j;
                break

    source_A.close();
    source_B.close();
    back_A.close();
    back_B.close()

    data_source = np.array(data_source);
    data_back = np.array(data_back)

    data_net = data_source.copy()
    # net count rate, corrected by the task 'nulccorr'
    data_net[:, 1] = data_source[:, 1] - data_back[:, 1] * backscal_factor
    # net count rate original
    data_net[:, 2] = data_source[:, 2] - data_back[:, 2] * backscal_factor

    locations = np.where(data_net[:, 1] > 0)
    data_net = data_net[locations]
    data_back = data_back[locations]
    lenth = len(data_net[:, 1])
    
    # calculate the intrinsic rms
    sigma_rms_intrinsic_2 = np.std(data_net[:, 2]) ** 2 - \
                            ((np.mean(data_net[:, 2] * data_net[:, 3]) * timebin) ** 0.5 / timebin) ** 2 \
                            - backscal_factor * (1 + backscal_factor) * \
                            ((np.mean(data_back[:, 2] * data_back[:, 3]) * timebin) ** 0.5 / timebin) ** 2

    # rescale the net lcs
    if sigma_rms_intrinsic_2 >= 0 and np.std(data_net[:, 2]) > 0:
        factor = (sigma_rms_intrinsic_2) ** 0.5 / np.std(data_net[:, 2])
        data_net[:, 1] = np.mean(data_net[:, 1]) + (data_net[:, 1] - np.mean(data_net[:, 1])) * factor
        data_net[:, 2] = np.mean(data_net[:, 2]) + (data_net[:, 2] - np.mean(data_net[:, 2])) * factor
    else:
        data_net[:, 1] = np.mean(data_net[:, 1])
        data_net[:int(lenth / 2), 1] += 1e-8
        data_net[:, 2] = np.mean(data_net[:, 2])

    # add flucuations to the lcs
    rands_background_in_src_reg = np.zeros((lenth, simulation_number))
    rands_background_in_bkg_reg = np.zeros((lenth, simulation_number))
    rands_source = np.zeros((lenth, simulation_number))
    for i in range(lenth):
        rands_background_in_src_reg[i, :] = np.random.poisson(
            data_back[i, 2] * data_back[i, 3] * backscal_factor * timebin, size=simulation_number) / (
                                                    timebin * data_back[i, 3])
        rands_background_in_bkg_reg[i, :] = np.random.poisson(data_back[i, 2] * data_back[i, 3] * timebin,
                                                              size=simulation_number) / (timebin * data_back[i, 3])
        rands_source[i, :] = np.random.poisson(data_net[i, 2] * data_net[i, 3] * timebin, size=simulation_number) / (
                timebin * data_net[i, 3])

    all_mean_back_in_src_reg = np.zeros(simulation_number)
    all_mean_back_in_bkg_reg = np.zeros(simulation_number)
    all_mean_source = np.zeros(simulation_number)
    all_instrinsic_net = np.zeros(simulation_number)

    for j in range(simulation_number):
        net_count_rate_origin = rands_source[:, j] + rands_background_in_src_reg[:, j] - rands_background_in_bkg_reg[:,
                                                                                         j] * backscal_factor
        net_count_rate = net_count_rate_origin * (data_net[:, 1] / data_net[:, 2])

        locations = np.where((net_count_rate >= key_count_rate[0]) & (net_count_rate < key_count_rate[1]))

        all_mean_back_in_src_reg[j] = np.sum(
            rands_background_in_src_reg[locations, j] * data_net[locations, 3]) / np.sum(
            data_net[locations, 3])

        all_mean_back_in_bkg_reg[j] = np.sum(
            rands_background_in_bkg_reg[locations, j] * data_net[locations, 3]) / np.sum(
            data_net[locations, 3])

        all_mean_source[j] = np.sum(rands_source[locations, j] * data_net[locations, 3]) / np.sum(
            data_net[locations, 3])

        all_instrinsic_net[j] = np.sum(data_net[locations, 2] * data_net[locations, 3]) / np.sum(
            data_net[locations, 3])

    source_correction = np.mean(all_mean_source) / np.mean(all_instrinsic_net)
    back_correction = np.mean(all_mean_back_in_bkg_reg) / np.mean(
        all_mean_back_in_src_reg) * source_correction * backscal_factor

    return (source_correction, back_correction)


# Input the NuSTAR obsid
ID = sys.argv[1]
'''
Input the cut count rates.
The input count rates should be the sum of the NuSTARDAS corrected ('rate' column in lc files instead of 'rate_orig'),
net count rates of FPMA and FPMB, which could be derived using the task 'lcmath'.
'''
key_rate = [float(sys.argv[2]), float(sys.argv[3])]

# Working directory. Supposing working in the products directory produced by the standard nuproducts task
directory = './'
# Used files, including 4 light curves and 2 spectra.
tails = ['A01_sr.lc', 'B01_sr.lc', 'A01_bk.lc', 'B01_bk.lc', 'A01_sr.pha', 'A01_bk.pha']
files = [directory + 'nu%s' % ID + x for x in tails]
#
results = perfrom_one_simulate(files, key_rate, simulation_number=10000)

print('Dealing with the NuSTAR observation: %s' % ID)
print('The corrections for the bins with count rates between [%s, %s]:' % (key_rate[0], key_rate[1]))
print('The \'AREASCAL\' headers of the source spectra should be multiplied by %.4f' % results[0])
print('The \'AREASCAL\' headers of the background spectra should be multiplied by %.4f' % results[1])

output_file = open('scale_factors.txt', 'a')
output_file.writelines('Dealing with the NuSTAR observation: %s \n' % ID)
output_file.writelines('The corrections for the bins with count rates between [%s, %s]:\n' % (key_rate[0], key_rate[1]))
output_file.writelines('The \'AREASCAL\' headers of the source spectra should be multiplied by %.4f\n' % results[0])
output_file.writelines(
    'The \'AREASCAL\' headers of the background spectra should be multiplied by %.4f\n\n' % results[1])
