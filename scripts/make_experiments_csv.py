import csv, os
OUT = os.path.expanduser("~/research/scripts/experiments_v2.csv")
MODELS = [
    dict(m="SABR", sfx="2_255", sel="C3_ACE_Net_SABR_cert_cifar10_2_255_v2.pt",
         ent="C3_ACE_Entropy_SABR_cert_cifar10_2_255_v2.pt",
         core="EB-0_cifar10_adv_2_255.pt", eps="0.00784313725"),
    dict(m="SABR", sfx="8_255", sel="C3_ACE_Net_SABR_cert_cifar10_8_255_v2.pt",
         ent="C3_ACE_Entropy_SABR_cert_cifar10_8_255_v2.pt",
         core="EB-0_cifar10_adv_8_255.pt", eps="0.03137254901"),
    dict(m="MTL", sfx="2_255", sel="C3_ACE_Net_MTL_cert_cifar10_2_255_v2.pt",
         ent="C3_ACE_Entropy_MTL_cert_cifar10_2_255_v2.pt",
         core="EB-0_cifar10_adv_2_255.pt", eps="0.00784313725"),
    dict(m="MTL", sfx="8_255", sel="C3_ACE_Net_MTL_cert_cifar10_8_255_v2.pt",
         ent="C3_ACE_Entropy_MTL_cert_cifar10_8_255_v2.pt",
         core="EB-0_cifar10_adv_8_255.pt", eps="0.03137254901"),
    dict(m="IBP", sfx="2_255", sel="C3_ACE_Net_IBP_cert_cifar10_2_255_v2.pt",
         ent="C3_ACE_Entropy_IBP_cert_cifar10_2_255_v2.pt",
         core="EB-0_cifar10_adv_2_255.pt", eps="0.00784313725"),
    dict(m="IBP", sfx="8_255", sel="C3_ACE_Net_IBP_cert_cifar10_8_255_v2.pt",
         ent="C3_ACE_Entropy_IBP_cert_cifar10_8_255_v2.pt",
         core="EB-0_cifar10_adv_8_255.pt", eps="0.03137254901"),
]
TAU_NET = [0.0, 0.3, 0.5, 0.7, 0.9]
TAU_ENT = [-0.1, -0.3, -0.5, -0.7, -0.9]
rows, eid = [], 41
for md in MODELS:
    s8 = "8" if md["sfx"] == "8_255" else ""
    for t in TAU_NET:
        rows.append([eid, "cifar10", md["eps"], md["core"], md["sel"], "C3_cifar10",
                     "C3_cifar10", "box", "net", str(t), f"C3_{md['m']}{s8}_SelNet_t{t}"])
        eid += 1
    for t in TAU_ENT:
        rows.append([eid, "cifar10", md["eps"], md["core"], md["ent"], "C3_cifar10",
                     "C3_cifar10", "box", "entropy", str(t), f"C3_{md['m']}{s8}_Ent_t{abs(t)}"])
        eid += 1
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id","dataset","eps","core","checkpoint","branch_net","gate_net",
                "cert_domain","gate_type","threshold","name"])
    w.writerow(["# CTRAIN-trained branches bridged into ACE; box domain; raw eps; ids 41-100"])
    w.writerows(rows)
print(f"wrote {OUT}: {len(rows)} rows (ids 41..{eid-1})")
