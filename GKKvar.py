###########################################################################
# Determination of the probability function of the X momentum in the tau
# restframe
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
# Use cases: Analysis of tau->e+invisible, mother particle mass measurement
#
# last modified: April 2020 -> release-04-0Y-0X
#
# 
# ##########################################################################
import sys
import os
import numpy as np
import pandas as pd
import ROOT
import concurrent.futures


def cosTheta(visible_tag_particle_E_CMSlist,
             visible_tag_particle_InvMlist,
             visible_tag_particle_p_CMSlist,
             beamEnergy=10.58,
             particle_mass=1.776
             ):
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
    ex, ey, ez, eE = signal_4momentum_vec_CMS
    e4P = ROOT.TLorentzVector(
        ex, ey, 
        ez, eE
    )
    t = ROOT.TLorentzVector(
        tx, ty, 
        tz, EcmsHalf
    )
    bv = t.BoostVector()
    e4P.Boost(-1 * direction * bv)
    return e4P.P()

def lorentzBoost(signal_4momentum_vec_CMS,
                 mother_particle_mom_CMS_list,
                 norm=True,
                 EcmsHalf=10.58 / 2,
                 particle_mass=1.776,
                 direction=-1
                 ):
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


def KumulativeDensityFct(signal_4vec_CMS_list_x_j,
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


def totalKumulativeDensityFct(signal_4vec_CMS_list_x,
                              signal_4vec_CMS_list_y,
                              signal_4vec_CMS_list_z,
                              signal_4vec_CMS_list_E,
                              visible_tag_particle_CMS_list_theta,
                              visible_tag_particle_CMS_list_phi,
                              cosThetaList,
                              cosThetaPrimeList,
                              DirAddVariables,
                              norm=True,
                              EcmsHalf=10.58 / 2,
                              particle_mass=1.776,
                              direction=-1,
                              ):
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
                EcmsHalf=EcmsHalf,
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
