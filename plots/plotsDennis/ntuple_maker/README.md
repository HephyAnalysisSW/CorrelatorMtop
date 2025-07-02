# Ntuple maker

This code produces a correlator Ntuple from NanoAOD.
The NanoAOD including PF candidates was produced using https://github.com/denschwarz/XtoYH.
There will be a more centrally produced NanoAOD soon.


The code produces a flat ROOT Tree, where each entry corresponds to a correlator triplet instead of event.
A simple event selection is implemented.
It performes also a matching of the correlators between detector and particle level.
For each triplet, following values are stored:

- `ievent`: Event number such that we can later identify which triplets come from the same event (float)
- `zeta_rec`: Value of the correlator zeta on detector level (float)
- `zeta_gen`: Value of the correlator zeta on particle level (float)
- `zeta_weight_rec`: Value of the associated correlator weight on detector level (float)
- `zeta_weight_gen`: Value of the associated correlator weight on particle level (float)
- `jetpt_rec`: Jet pt on detector level (float)
- `jetpt_gen`: Jet pt on particle level (float)
- `has_rec_info`: Does the correlator exist on detector level (int)
- `has_gen_info`: Does the correlator exist on particle level (int)
- `pass_triplet_top_gen`: Does the correlator pass the requirements of the top correlator on detector level (int)
- `pass_triplet_top_rec`: Does the correlator pass the requirements of the top correlator on particle level (int)
- `BW_reweight_171p5`: Breit-Wigner weight to reweight to mtop = 171.5 GeV (float)
- `BW_reweight_173p5`: Breit-Wigner weight to reweight to mtop = 173.5 GeV (float)
- `mtop`: Parton level mtop (float)
- `event_weight_gen`: Particle level event weight (float)
- `event_weight_rec`: Detector level event weight (float)
