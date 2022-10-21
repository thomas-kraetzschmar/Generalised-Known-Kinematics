import ROOT
import concurrent.futures

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