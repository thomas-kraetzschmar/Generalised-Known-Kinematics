###########################################################################
# Determination of the probability function of the X momentum in e.g. the tau rest frame
# 
#
#                  X +...
#                  |
#        e+ e- -> tau+ tau-
#                       |
#                     h h h nu
#
# Contributors:
# Thomas Kraetzschmar
#
# Use cases: Analysis of tau->e+invisible, tau mass measurement
#
# Offical code for the GKK method described in: https://arxiv.org/abs/2109.14455
# 
# CMS: center of mass system of the event 
###########################################################################
import concurrent.futures
import GKK_functions.py

def totalDensityFct(signal_4vec_CMS_list_x,
                    signal_4vec_CMS_list_y,
                    signal_4vec_CMS_list_z,
                    signal_4vec_CMS_list_E,
                    visible_tag_particle_CMS_list_theta,
                    visible_tag_particle_CMS_list_phi,
                    cosThetaList,
                    cosThetaPrimeList,
                    DirAddVariables,
                    norm=True,
                    particle_energy=10.58 / 2,
                    particle_mass=1.776,
                    direction=-1,
                    ):
    '''
    Sample the density function of the signal momentum in the restframe of the mother for all events.

    input:
    - signal_4vec_CMS_list_x: x component of the signal momentum in the CMS frame
    - signal_4vec_CMS_list_y: y component of the signal momentum in the CMS frame
    - signal_4vec_CMS_list_z: z component of the signal momentum in the CMS frame
    - signal_4vec_CMS_list_E: Energy in the CMS frame of the signal
    - visible_tag_particle_CMS_list_theta: Theta angle of the visible particle system of the tag side in the coordinate system of the detector in the CMS frame
    - visible_tag_particle_CMS_list_phi: Phi angle of the visible particle system of the tag side in the coordinate system of the detector in the CMS frame
    - cosThetaList: Angle between the momentum of the mother particle of the tag particles and the momentum of the visible particles system.
    - cosThetaPrimeList: Angle between the momentum of the mother particle of the signal particles and the momentum of the (detected) particles.
    - DirAddVariables: Additional event properties stored in a directory format
    - norm=True: Indicater if mother_particle_mom_CMS_list is a normalised vector. Default is a normalised vector
    - particle_energy==10.58 / 2 [GeV]: Half of the beam energy -- the total energy of the mother particle. Default is the beam energy (in GeV) of colliders with Upsilon 4S resonance beam energy
    - particle_mass=1.776 [GeV/c^2]: rest mass of the mother particle under consideration, e.g. the tau particle. Default is the tau mass
    - direction=-1: sign of the referencee frame transformation (this is need becaus in particle pair events we reconstruct the tag particl's momentum, which has the opposite flight direction than the signal particle). Default is the opposite direction as is the case in particle pair events.

    output:
    - gkkDf: list of the signal momentum in the restframe for all mother momenta -- the sampled density function of the signal particl's momentum in the restframe of the mother -- as a pandas dataframe. The pandas dataframe formate allows to propagate additional event properties for every entry.
    '''
    signal_4vec_CMS_list_theta = np.arctan(
        np.divide(
            np.sqrt(
                np.add(
                    np.power(signal_4vec_CMS_list_x, 2),
                    np.power(signal_4vec_CMS_list_y, 2)
                )
            ), 
            signal_4vec_CMS_list_z
        )
    )
    signal_4vec_CMS_list_phi = np.arctan( np.divide(signal_4vec_CMS_list_y, signal_4vec_CMS_list_x) )
    signal_4vec_CMS_list_p = np.sqrt(
        np.add(
            np.add(
                np.power(signal_4vec_CMS_list_x, 2), 
                np.power(signal_4vec_CMS_list_y, 2)
            ),
            np.power(signal_4vec_CMS_list_z, 2)
        )
    )
    # Avoide issues in iteration process, create iteration list 
    if type(signal_4vec_CMS_list_x) == pd.core.series.Series:
        print('pd dataframe')
        iterator = signal_4vec_CMS_list_x.keys()
    else:
        iterator = range(len(signal_4vec_CMS_list_x))

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_u = {
            executor.submit(
                KumulativeDensityFct,
                signal_4vec_CMS_list_x[j],
                signal_4vec_CMS_list_y[j],
                signal_4vec_CMS_list_z[j],
                signal_4vec_CMS_list_E[j],
                visible_tag_particle_CMS_list_theta[j],
                visible_tag_particle_CMS_list_phi[j],
                cosThetaList[j],
                signal_4vec_CMS_list_theta[j],
                signal_4vec_CMS_list_phi[j],
                cosThetaPrimeList[j],
                [DirAddVariables[k][j] for k in DirAddVariables],
                DirAddVariableKeys,
                norm=norm,
                particle_energy=particle_energy,
                particle_mass=particle_mass,
                direction=direction): j
            for j in iterator
        }
        signal_momentum_List = [
            future.result()
            for future in concurrent.futures.as_completed(future_u)
        ]
    gkkDf = pd.concat(signal_momentum_List)
    gkkDf = gkkDf.reset_index(drop=True)

    return gkkDf