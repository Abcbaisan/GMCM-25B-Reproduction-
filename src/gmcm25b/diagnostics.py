"""Descriptive overlap checks; channel energy equality does not imply CSI equality."""
import hashlib
import numpy as np

def audit_overlap(train,valid):
    exact=[];energy=[];splits={}
    for name,d in [("train",train),("valid",valid)]:
        x=np.asarray(d.sinr,dtype="<f8")
        exact_hash=[hashlib.sha256(row.tobytes()).hexdigest() for row in x]
        nf=d.meta.noise_floor.to_numpy(dtype=float)
        # Invert the declared signal_scale=.001, epsilon=0 preprocessing.
        power=x*(10**((nf-30)/10)/122)[:,None]/.001
        rounded_hash=[hashlib.sha256(",".join(format(float(v),".11g") for v in row).encode()).hexdigest()
                      for row in power]
        exact.append(exact_hash);energy.append(rounded_hash)
        splits[name]={"rows":len(x),"exact_sinr_unique_rows":len(set(exact_hash)),
                      "channel_energy_unique_rows_11_significant_digits":len(set(rounded_hash))}
    training_energy=set(energy[0])
    return {"method":"Exact float64 SINR row SHA256, plus total channel energy reconstructed using declared physical scaling and rounded to 11 significant decimal digits.",
            "splits":splits,
            "shared_exact_sinr_rows":len(set(exact[0])&set(exact[1])),
            "shared_rounded_energy_vectors":len(set(energy[0])&set(energy[1])),
            "valid_rows_with_rounded_energy_seen_in_train":sum(h in training_energy for h in energy[1]),
            "limits":"Energy vectors sum antenna powers and discard complex phase/spatial information. Approximate energy repetition is not proof of identical CSI or label leakage. Exact SINR non-overlap does not rule out approximate input similarity."}
