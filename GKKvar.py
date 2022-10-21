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
import sys
import os
import numpy as np
import pandas as pd
import ROOT


def cosTheta(visible_tag_particle_E_CMSlist,
             visible_tag_particle_InvMlist,
             visible_tag_particle_p_CMSlist,
             beamEnergy=10.58,
             particle_mass=1.776
             ):
    '''
    Calculate the cosine of the angle between the combination of the visible (daughter) particles' momentum and the mother particle's momentum 

    input:
    - visible_tag_particle_E_CMSlist: Energy of the visible particle system in the CMS 
    - visible_tag_particle_InvMlist: invariant mass -- "rest mass" -- of the visible particle system in the CMS 
    - visible_tag_particle_p_CMSlist: absolute value of the momentum for the visible particle system in the CMS 
    - beamEnergy: collision energy of particles in the CMS
    - particle_mass: rest mass of the mother particle under consideration, e.g. the tau particle.

    output:
    - cosTheta: Angle between the momentum of the mother particle under consideration and the moentum of the visible particles system.
    '''
    abs_mom_mother_particle = np.sqrt(
        np.subtract(
            np.power(beamEnergy / 2, 2), 
            np.power(particle_mass, 2)
        )
    )
    cosTheta = np.divide(
        np.subtract(
            np.multiply(beamEnergy, visible_tag_particle_E_CMSlist),
            np.add(
                np.power(particle_mass, 2), 
                np.power(visible_tag_particle_InvMlist, 2)
            )
        ),
        np.multiply(
            2, 
            np.multiply(
                abs_mom_mother_particle, 
                visible_tag_particle_p_CMSlist
            )
        )
    )
    return cosTheta


def mother_particle_vec_list(thetaTag, 
                             phiTag, 
                             cosTheta_visible_tag_particle_momentum, 
                             phiL, 
                             mother_particle_p=1
                             ):
    '''
    Generate list of mother particle vectors

    input:
    - thetaTag: Theta angle of the visible particle system in the coordinate system of the detector
    - phiTag: Phi angle of the visible particle system in the coordinate system of the detector
    - cosTheta_visible_tag_particle_momentum: Angle between the momentum of the mother particle under consideration and the momentum of the visible particles system.

    output:
    - three component list being the lists of the vector components of the mother vector
    '''
    sinTheta_visible_tag_particle_momentum = np.sqrt(1 - cosTheta_visible_tag_particle_momentum**2)
    sinTheta = np.sin(thetaTag)
    cosTheta = np.cos(thetaTag)
    sinPhi = np.sin(phiTag)
    cosPhi = np.cos(phiTag)
    cpi = np.cos(phiL)
    spi = np.sin(phiL)
    mother_Vec_list_X = np.multiply(
        mother_particle_p,
        np.add(
            cosTheta_visible_tag_particle_momentum * sinTheta * cosPhi,
            np.subtract(
                np.multiply(
                    sinTheta_visible_tag_particle_momentum * cosPhi * cosTheta, 
                    cpi
                ),
                np.multiply(
                    sinTheta_visible_tag_particle_momentum * sinPhi, 
                    spi
                )
            )
        )
    )
    mother_Vec_list_Y = np.multiply(
        mother_particle_p,
        np.add(
            cosTheta_visible_tag_particle_momentum * sinTheta * sinPhi,
            np.add(
                np.multiply(sinTheta_visible_tag_particle_momentum * sinPhi * cosTheta, cpi),
                np.multiply(sinTheta_visible_tag_particle_momentum * cosPhi, spi)
            )
        )
    )
    mother_Vec_list_Z = np.multiply(
        mother_particle_p,
        np.add(
            cosTheta_visible_tag_particle_momentum * cosTheta,
            np.multiply(
                sinTheta_visible_tag_particle_momentum * (-sinTheta), 
                cpi
            )
        )
    )
    return [mother_Vec_list_X, mother_Vec_list_Y, mother_Vec_list_Z]


def boostvec(tx, 
             ty, 
             tz, 
             EcmsHalf, 
             signal_4momentum_vec_CMS, 
             direction
             ):
    '''
    Generate ROOT 4-vectors of the signal particle and the boost vector -- the direction and magnitude of the reference frame transformation.

    input:
    - tx: x component of the mother particle momentum
    - ty: y component of the mother particle momentum
    - tz: z component of the mother particle momentum
    - EcmsHalf: Half of the beam energy -- the total energy of the mother particle
    - signal_4momentum_vec_CMS: 4 momentum vector components of the signal particle
    - direction: sign of the referencee frame transformation (this is need becaus in particle pair events we reconstruct the tag particl's momentum, which has the opposite flight direction than the signal particle) 

    output:
    - ROOT 4 vector object
    '''
    sig_x, sig_y, sig_z, sig_E = signal_4momentum_vec_CMS
    sig_4mom = ROOT.TLorentzVector(
        sig_x, sig_y, 
        sig_z, sig_E
    )
    t = ROOT.TLorentzVector(
        tx, ty, 
        tz, EcmsHalf
    )
    boost_vector = t.BoostVector()
    sig_4mom.Boost(-1 * direction * boost_vector)
    return sig_4mom.P()

def lorentzBoost(signal_4momentum_vec_CMS,
                 mother_particle_mom_CMS_list,
                 norm=True,
                 EcmsHalf=10.58 / 2,
                 particle_mass=1.776,
                 direction=-1
                 ):
    '''
    boost signal particle momentum into restframe of the mother particle using the reconstructed mother momenta

    input:
    - signal_4momentum_vec_CMS: 4 momentum of the signal particle
    - mother_particle_mom_CMS_list: list of the cartesian components of the tag mother particle momenta
    - norm=True: Indicater if mother_particle_mom_CMS_list is a normalised vector. Default is a normalised vector
    - EcmsHalf==10.58 / 2 [GeV]: Half of the beam energy -- the total energy of the mother particle. Default is the beam energy (in GeV) of colliders with Upsilon 4S resonance beam energy
    - particle_mass=1.776 [GeV/c^2]: rest mass of the mother particle under consideration, e.g. the tau particle. Default is the tau mass
    - direction=-1: sign of the referencee frame transformation (this is need becaus in particle pair events we reconstruct the tag particl's momentum, which has the opposite flight direction than the signal particle). Default is the opposite direction as is the case in particle pair events.

    output:
    - list of the signal momentum for the rest frames of all reconstructed tag mother momenta
    '''
    tx, ty, tz = mother_particle_mom_CMS_list
    if norm:
        tP = np.sqrt(EcmsHalf**2 - particle_mass**2)
        tx = np.multiply(tP, tx)
        ty = np.multiply(tP, ty)
        tz = np.multiply(tP, tz)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_u = {
            executor.submit(
                boostvec, tx[i], 
                ty[i], tz[i], 
                EcmsHalf, signal_4momentum_vec_CMS,
                direction
            ): i
            for i in range(len(tx))
        }
    return [
        future.result()
        for future in concurrent.futures.as_completed(future_u)
    ]

def goodPhiRegion(tag_theta_cms, 
                  tag_phi_cms, 
                  sig_theta_cms, 
                  sig_phi_cms,
                  cosTheta, 
                  cosThetaPrim
                  ):
    '''
    As explained in the paper referenced above, due to smearing it is possible to reconstruct mother momenta which are unphysical. This function determined the physical mother momenta (good momenta). The Determination of good mother momenta can be reduced to determining those values of the phi angle which are physical, as described in the linked publication.

    input:
    - thetaTag: Theta angle of the visible particle system of the tag side in the coordinate system of the detector
    - tag_phi_cms: Phi angle of the visible particle system of the tag in the coordinate system of the detector
    - sig_theta_cms: Theta angle of the visible particle system of the signal side in the coordinate system of the detector
    - sig_phi_cms: Phi angle of the visible particle system of the signal in the coordinate system of the detector
    - cosTheta: Angle between the momentum of the mother particle of the tag particles and the momentum of the visible particles system.
    - cosThetaPrim: Angle between the momentum of the mother particle of the signal particle(s) and the momentum of the visible particles system.

    output:
    - GoodSample: list of Boolian values indicating which mother momenta are physical
    '''
    phiList = np.arange(
        0, 
        2 * np.pi, 
        2 * np.pi / 1000
    )
    tag_mother = mother_particle_vec_list(
        tag_theta_cms, 
        tag_phi_cms, 
        cosTheta, 
        phiList
    )
    for i in range(3):
        tag_mother[i] = np.multiply(tag_mother[i], -1)

    signal_mother = mother_particle_vec_list(
        sig_theta_cms,
        sig_phi_cms, 
        cosThetaPrim, 
        phiList
    )
    # find signal mother particle range
    signal_motherMax = [np.amax(signal_mother[i]) for i in range(3)]
    signal_motherMin = [np.amin(signal_mother[i]) for i in range(3)]

    # Check if tag signal_mother are in signal mother particle range
    tagLessEqualS = [np.less_equal(tag_mother[i], signal_motherMax[i]) for i in range(3)]
    tagMoreEqualS = [np.less_equal(signal_motherMin[i], tag_mother[i]) for i in range(3)]
    r = np.multiply(tagLessEqualS, tagMoreEqualS)
    GoodSample1 = np.multiply(
        r[0], 
        np.multiply(r[1], r[2])
    )
    signal_mother_dxyz = np.divide(
        np.subtract(signal_motherMax, signal_motherMin), 
        2
    )
    signal_mothermid = np.add(signal_mother_dxyz, signal_motherMin)
    signal_mother_r = np.sqrt(sum([i**2 for i in signal_mother_dxyz]))
    tag_mother_r = np.sqrt(
        np.add(
            np.add(
                np.power(
                    np.subtract(tag_mother[0], signal_mothermid[0]), 
                    2
                ),
                np.power(
                    np.subtract(tag_mother[1], signal_mothermid[1]), 
                    2
                )
            ),
            np.power(
                np.subtract(tag_mother[0], signal_mothermid[0]), 
                2
            )
        )
    )
    GoodSample2 = np.less_equal(tag_mother_r, signal_mother_r)
    GoodSample = np.multiply(GoodSample1, GoodSample2)

    return GoodSample

def sampleInGoodPhiRegion(GoodSample):
    '''
    After determining the physical momenta region to retain the weight of every event equal, this function resamples the same number mother momenta for the physical momentum space

    input:
    - GoodSample: list of Boolian values indicating which mother momenta are physical

    output:
    - return a set of physical mother momenta
    '''
    phiList = np.arange(
        0, 
        2 * np.pi, 
        2 * np.pi / 1000
    )
    goodAngle = np.multiply(phiList, GoodSample)
    goodAngleGrad = goodAngle[1:] - goodAngle[:-1]

    # Find phi in which tag and signal mother particle regions intersect
    endPoint = np.where(goodAngleGrad < 0)[0]
    startpoint = np.where(goodAngleGrad > 2 * np.pi / 1000 * 1.5)[0]

    # include start and enpoint of sample (0, 2pi)
    try:
        if len(endPoint) == len(startpoint) and endPoint[0] < startpoint[0]:
            startpoint = np.insert(startpoint, 0, 0, axis=0)
            endPoint = np.append(endPoint, 999)
        elif (len(endPoint) != len(startpoint)) and (endPoint[-1] <
                                                     startpoint[-1]):
            endPoint = np.append(endPoint, 999)

        elif (len(endPoint) != len(startpoint)) and (endPoint[0] <
                                                     startpoint[0]):
            startpoint = np.insert(startpoint, 0, 0, axis=0)

        else:
            pass

    except IndexError:
        startpoint = np.array([0])
        endPoint = np.array([999])

    fractionL = np.subtract(endPoint, startpoint)
    fraction = np.sum(fractionL)
    percent = np.multiply([round(i / fraction, 2) for i in fractionL], 1000)

    GoodMotherParticleAngles = []
    ang_sample = []

    for i in range(len(startpoint)):
        start = phiList[startpoint[i]]
        end = phiList[endPoint[i]]
        delta = end - start
        ang_i = np.arange(start, end, delta / percent[i])
        ang_sample.append(ang_i)

    if len(ang_sample) == 1:
        GoodMotherParticleAngles = ang_sample[0]
    elif len(ang_sample) == 0:
        print('ERROR: Somthing went wrong with the mother-particle-vector determination')
        print(startpoint.size, endPoint.size)
    else:
        GoodMotherParticleAngles = np.concatenate(ang_sample, axis=0)

    return GoodMotherParticleAngles


def DensityFct(signal_4vec_CMS_list_x_j,
               signal_4vec_CMS_list_y_j,
               signal_4vec_CMS_list_z_j,
               signal_4vec_CMS_list_E_j,
               visible_tag_particle_CMS_list_theta_j,
               visible_tag_particle_CMS_list_phi_j,
               cosThetaList_j,
               signal_4vec_CMS_list_theta_j,
               signal_4vec_CMS_list_phi_j,
               cosThetaPrimeList_j,
               DirAddVariableValues,
               DirAddVariableKeys,
               norm=True,
               EcmsHalf=10.58 / 2,
               particle_mass=1.776,
               direction=-1,
               ):
    '''
    Sample the density function of the signal momentum in the restframe of the mother for one event.

    input:
    - signal_4vec_CMS_list_x_j: x component of the signal momentum in the CMS frame for one event
    - signal_4vec_CMS_list_y_j: y component of the signal momentum in the CMS frame for one event
    - signal_4vec_CMS_list_z_j: z component of the signal momentum in the CMS frame for one event
    - signal_4vec_CMS_list_E_j: Energy in the CMS frame of the signal for one event
    - visible_tag_particle_CMS_list_theta_j: Theta angle of the visible particle system of the tag side in the coordinate system of the detector in the CMS frame for one event
    - visible_tag_particle_CMS_list_phi_j: Phi angle of the visible particle system of the tag side in the coordinate system of the detector in the CMS frame for one event
    - cosThetaList_j: Angle between the momentum of the mother particle of the tag particles and the momentum of the visible particles system for one event.
    - signal_4vec_CMS_list_theta_j: Theta angle of the  particle of the signal side in the coordinate system of the detector in the CMS frame for one event
    - signal_4vec_CMS_list_phi_j: Phi angle of the  particle of the signal side in the coordinate system of the detector in the CMS frame for one event
    - cosThetaPrimeList_j: Angle between the momentum of the mother particle of the signal particles and the momentum of the (detected) particles for one event.
    - DirAddVariableValues: Additional event property value 
    - DirAddVariableKeys: Additional event property directory key
    - norm=True: Indicater if mother_particle_mom_CMS_list is a normalised vector. Default is a normalised vector
    - EcmsHalf==10.58 / 2 [GeV]: Half of the beam energy -- the total energy of the mother particle. Default is the beam energy (in GeV) of colliders with Upsilon 4S resonance beam energy
    - particle_mass=1.776 [GeV/c^2]: rest mass of the mother particle under consideration, e.g. the tau particle. Default is the tau mass
    - direction=-1: sign of the referencee frame transformation (this is need becaus in particle pair events we reconstruct the tag particl's momentum, which has the opposite flight direction than the signal particle). Default is the opposite direction as is the case in particle pair events.

    output:
    - gkkDfMc: list of the signal momentum in the restframe for all mother momenta -- the sampled density function of the signal particl's momentum in the restframe of the mother -- as a pandas dataframe. The pandas dataframe formate allows to propagate additional event properties for every entry.
    '''
    signal_4momentum_vec_CMS = [
        signal_4vec_CMS_list_x_j,
        signal_4vec_CMS_list_y_j,
        signal_4vec_CMS_list_z_j,
        signal_4vec_CMS_list_E_j,
    ]
    goodPhis = goodPhiRegion(visible_tag_particle_CMS_list_theta_j, visible_tag_particle_CMS_list_phi_j,
                             signal_4vec_CMS_list_theta_j, signal_4vec_CMS_list_phi_j,
                             cosThetaList_j, cosThetaPrimeList_j)
    phiList = sampleInGoodPhiRegion(goodPhis)
    n_mother_particle_mom_CMS_list = mother_particle_vec_list(visible_tag_particle_CMS_list_theta_j, visible_tag_particle_CMS_list_phi_j,
                                 cosThetaList_j, phiList)
    peTrue = lorentzBoost(
        signal_4momentum_vec_CMS,
        n_mother_particle_mom_CMS_list,
        norm=norm,
        EcmsHalf=EcmsHalf,
        particle_mass=particle_mass,
        direction=direction,
    )
    dMC = {'GKK': peTrue}
    gkkDfMc = pd.DataFrame(data=dMC)
    lengthStatistics = len(peTrue)
    for r in range(len(DirAddVariableValues)):
        gkkDfMc[DirAddVariableKeys[r]] = np.ones(
            lengthStatistics) * DirAddVariableValues[r]

    return gkkDfMc


